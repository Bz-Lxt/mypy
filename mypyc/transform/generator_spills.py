"""Move generator registers that survive suspension onto the generator frame."""

from __future__ import annotations

from mypyc.analysis.dataflow import analyze_live_regs_details_with_exception_edges, cleanup_cfg
from mypyc.common import TEMP_ATTR_NAME, generator_frame_attribute_prefix
from mypyc.ir.class_ir import ClassIR
from mypyc.ir.func_ir import FuncIR
from mypyc.ir.ops import (
    NO_TRACEBACK_LINE_NO,
    Assign,
    AssignMulti,
    BaseAssign,
    BasicBlock,
    Branch,
    GetAttr,
    LoadAddress,
    LoadErrorValue,
    Op,
    Register,
    SetAttr,
    Value,
)
from mypyc.ir.rtypes import RType, is_simple_refcounted_pointer, object_rprimitive
from mypyc.namegen import exported_name


def promote_generator_registers(ir: FuncIR, frame: ClassIR) -> None:
    """Replace registers live across a suspension with frame attributes.

    This runs before uninitialized-value, exception, and refcount transforms.
    """
    cleanup_cfg(ir.blocks)
    nullable = nullable_registers(ir.blocks)
    live_in, live_after = analyze_live_regs_details_with_exception_edges(ir.blocks)
    registers = promotable_registers(ir.blocks, live_in[ir.blocks[0]])
    if not registers:
        return

    interference = register_interference(registers, live_after)
    slots = allocate_slots(frame, registers, interference)
    rewrite_registers(ir.blocks, ir.arg_regs[0], slots, nullable)

    # Keep this close to the rewrite: it catches a new op kind that is not
    # handled above without restating the liveness analysis.
    for block in ir.blocks:
        for op in block.ops:
            assert not any(source in slots for source in op.sources())
            assert not (isinstance(op, BaseAssign) and op.dest in slots)


def registers_live_across_yield(ir: FuncIR) -> list[Register]:
    """Return promotable registers live at generator-helper entry."""
    live_in, _ = analyze_live_regs_details_with_exception_edges(ir.blocks)
    return promotable_registers(ir.blocks, live_in[ir.blocks[0]])


def promotable_registers(blocks: list[BasicBlock], live: set[Value]) -> list[Register]:
    """Select promotable registers from a live set and order them deterministically."""
    excluded = unpromotable_registers(blocks)
    registers = [
        value
        for value in live
        if isinstance(value, Register) and not value.is_arg and value not in excluded
    ]
    return order_registers(registers, blocks)


def register_interference(
    registers: list[Register], live_after: dict[tuple[BasicBlock, int], set[Value]]
) -> dict[Register, set[Register]]:
    """Build the interference graph of registers that need distinct frame slots.

    A register defined at an operation interferes with every register live after
    that operation, including across branches, error edges and yield
    continuations (all already present in live_after). Two registers sharing a
    slot without an edge between them can never observe each other's stored
    value, so reusing the storage is safe through errors, generator close and
    early completion.
    """
    candidates = set(registers)
    edges: dict[Register, set[Register]] = {register: set() for register in registers}
    for (block, index), live in live_after.items():
        op = block.ops[index]
        if not isinstance(op, Assign) or not isinstance(op.dest, Register):
            continue
        dest = op.dest
        if dest not in candidates:
            continue
        for register in live:
            if register is not dest and isinstance(register, Register) and register in candidates:
                edges[dest].add(register)
                edges[register].add(dest)
    return edges


def unpromotable_registers(blocks: list[BasicBlock]) -> set[Register]:
    """Find registers whose representation requires a real C local."""
    result: set[Register] = set()
    for block in blocks:
        for op in block.ops:
            if isinstance(op, LoadAddress) and isinstance(op.src, Register):
                result.add(op.src)
            elif isinstance(op, AssignMulti):
                result.add(op.dest)
            elif (
                isinstance(op, Assign)
                and isinstance(op.src, LoadErrorValue)
                and op.src.undefines
                and op.dest.type.error_overlap
            ):
                result.add(op.dest)
    return result


def nullable_registers(blocks: list[BasicBlock]) -> set[Register]:
    """Find registers for which the error value is an ordinary state."""
    result: set[Register] = set()
    for block in blocks:
        for op in block.ops:
            if (
                isinstance(op, Branch)
                and op.op == Branch.IS_ERROR
                and isinstance(op.value, Register)
            ):
                result.add(op.value)
            elif (
                isinstance(op, Assign)
                and isinstance(op.src, LoadErrorValue)
                and not op.src.undefines
            ):
                result.add(op.dest)
    return result


def order_registers(registers: list[Register], blocks: list[BasicBlock]) -> list[Register]:
    """Order registers by first appearance for deterministic frame layouts."""
    positions: dict[Register, int] = {}
    for block in blocks:
        for op in block.ops:
            values = op.sources()
            if isinstance(op, BaseAssign):
                values = [op.dest, *values]
            for value in values:
                if isinstance(value, Register) and value not in positions:
                    positions[value] = len(positions)
    return sorted(registers, key=positions.__getitem__)


def can_share_slot_type(left: RType, right: RType) -> bool:
    """Check whether two values can be stored in the same frame attribute.

    Identical representations always match. Distinct types can share one
    'PyObject *' field, since the storage layout, ownership rules and the NULL
    undefined/error value are identical for every reference-counted pointer.
    Other unboxed representations (tagged ints, floats, bools, inline tuples)
    keep separate slots.
    """
    if left == right:
        return True
    return is_simple_refcounted_pointer(left) and is_simple_refcounted_pointer(right)


class FrameSlot:
    """A frame attribute shared by one or more non-interfering registers."""

    def __init__(self, first: Register, index: int) -> None:
        self.members = [first]
        self.index = index
        self.type = first.type

    def can_add(self, register: Register, interference: dict[Register, set[Register]]) -> bool:
        if not can_share_slot_type(self.type, register.type):
            return False
        return all(register not in interference[member] for member in self.members)

    def add(self, register: Register) -> None:
        self.members.append(register)
        if self.type != register.type:
            # The members agree on a single, reference-counted 'PyObject *'
            # representation, so the shared field uses the generic type.
            assert is_simple_refcounted_pointer(self.type)
            assert is_simple_refcounted_pointer(register.type)
            self.type = object_rprimitive


def allocate_slots(
    frame: ClassIR, registers: list[Register], interference: dict[Register, set[Register]]
) -> dict[Register, str]:
    """Assign registers to frame attributes, reusing slots when safe.

    Registers are processed in deterministic first-appearance order and placed
    into the earliest compatible slot, so frame layouts for identical input are
    stable across compilations.
    """
    owner = exported_name(frame.fullname)
    source_prefix = generator_frame_attribute_prefix(
        frame.fullname, is_final_class=frame.is_final_class
    )
    groups: list[FrameSlot] = []
    for register in registers:
        for slot in groups:
            if slot.can_add(register, interference):
                slot.add(register)
                break
        else:
            groups.append(FrameSlot(register, len(groups)))

    slots: dict[Register, str] = {}
    for slot in groups:
        first = slot.members[0]
        if first.name:
            name = available_attr_name(frame, source_prefix + first.name)
        else:
            name = f"{TEMP_ATTR_NAME}3_{owner}_{slot.index}"
            if all(not member.name and slot.type.error_overlap for member in slot.members):
                # Compiler temporaries are assigned before use. Recording a
                # default avoids allocating an undefinedness bitmap for them.
                frame.attrs_with_defaults.add(name)
        frame.attributes[name] = slot.type
        for member in slot.members:
            slots[member] = name
    return slots


def available_attr_name(frame: ClassIR, base: str) -> str:
    if not frame.has_attr(base):
        return base
    suffix = 2
    while frame.has_attr(f"{base}__{suffix}"):
        suffix += 1
    return f"{base}__{suffix}"


def rewrite_registers(
    blocks: list[BasicBlock],
    frame_reg: Register,
    slots: dict[Register, str],
    nullable: set[Register],
) -> None:
    """Replace promoted register accesses with generator frame attribute accesses."""
    # IRTransform maps each value to one replacement, but mutable registers need a fresh
    # attribute read at each use, so we can't use IRTransform here.
    for block in blocks:
        old_ops = block.ops
        block.ops = []
        for op in old_ops:
            replacements: dict[Register, GetAttr] = {}
            new_sources: list[Value] = []
            for source in op.sources():
                if isinstance(source, Register) and source in slots:
                    read = replacements.get(source)
                    if read is None:
                        read = GetAttr(
                            frame_reg,
                            slots[source],
                            source_line(op, source),
                            rtype=source.type,
                            allow_error_value=source in nullable,
                        )
                        block.ops.append(read)
                        replacements[source] = read
                    new_sources.append(read)
                else:
                    new_sources.append(source)
            op.set_sources(new_sources)

            if isinstance(op, Assign) and op.dest in slots:
                block.ops.append(
                    SetAttr(frame_reg, slots[op.dest], op.src, source_line(op, op.dest))
                )
            else:
                block.ops.append(op)


def source_line(op: Op, register: Register) -> int:
    if op.line >= 0 or op.line == NO_TRACEBACK_LINE_NO:
        return op.line
    if register.line >= 0:
        return register.line
    return NO_TRACEBACK_LINE_NO
