const std = @import("std");
const builtin = @import("builtin");

pub fn UnionInit(comptime Union: type) fn (comptime std.meta.Tag(Union), anytype) Union {
    return struct {
        pub fn init(comptime tag: std.meta.Tag(Union), payload: std.meta.TagPayload(Union, tag)) Union {
            return @unionInit(Union, @tagName(tag), payload);
        }
    }.init;
}
const Value = u64;
const VMRegister = usize;
const VMLocal = usize;

const LessThan = struct {
    lhs: VMRegister,
};
const JumpConditional = struct {
    true_target: *Block,
    false_target: *Block,
};

const Instruction = union(enum) {
    LoadImmediate: Value,
    Load: VMRegister,
    Store: VMRegister,
    SetLocal: VMLocal,
    GetLocal: VMLocal,
    Increment: void,
    LessThan: LessThan, // lhs
    Jump: *Block,
    JumpConditional: JumpConditional,
    Exit: void,

    pub const init = UnionInit(@This());

    pub fn dump(self: @This(), writer: anytype) !void {
        try writer.print("{d}=", .{@intFromEnum(std.meta.activeTag(self))});
        try writer.print("{}", .{self});
    }
};

const VM = struct {
    const Registers = std.ArrayList(Value);
    const Locals = std.ArrayList(Value);

    registers: Registers,
    locals: Locals,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !@This() {
        var vm: VM = .{
            .allocator = allocator,
            .registers = try Registers.initCapacity(allocator, 8),
            .locals = try Locals.initCapacity(allocator, 12),
        };
        vm.registers.expandToCapacity();
        vm.locals.expandToCapacity();
        return vm;
    }

    pub inline fn setLocal(self: *@This(), i: usize, val: Value) !void {
        try self.locals.ensureTotalCapacity(i + 1);
        self.locals.expandToCapacity();
        self.locals.items[i] = val;
    }

    pub fn deinit(self: *@This()) void {
        self.registers.deinit();
        self.locals.deinit();
    }

    pub fn dump(self: *@This(), writer: anytype) !void {
        try writer.print("Registers {d}:\n", .{self.registers.items.len});
        for (self.registers.items, 0..) |reg, i| {
            try writer.writeByteNTimes(' ', 4);
            try writer.print("{d}: {d}\n", .{ i, reg });
        }

        try writer.print("Locals {d}:\n", .{self.locals.items.len});
        for (self.locals.items, 0..) |loc, i| {
            try writer.writeByteNTimes(' ', 4);
            try writer.print("{d}: {d}\n", .{ i, loc });
        }

        _ = try writer.write("\n");
    }
};

const Block = struct {
    const Instructions = std.ArrayList(Instruction);
    const Jumps = std.ArrayList(usize);

    instructions: Instructions,
    // Offsets into the instruction stream where we have RIP-relative jump offsets to here that need patching.
    jumps: Jumps,
    // Offset where this block starts
    offset: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{
            .instructions = Instructions.init(allocator),
            .jumps = Jumps.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.instructions.deinit();
        self.jumps.deinit();
    }

    pub fn dump(self: *@This(), writer: anytype, spaces: usize) !void {
        for (self.instructions.items, 0..) |instruction, i| {
            _ = try writer.writeByteNTimes(' ', spaces);
            try writer.print("Instrution {d}: ", .{i});
            try instruction.dump(writer);
            _ = try writer.write("\n");
        }
    }

    pub inline fn append(self: *@This(), i: Instruction) !void {
        try self.instructions.append(i);
    }

    pub inline fn appendInit(self: *@This(), comptime tag: std.meta.Tag(Instruction), v: anytype) !void {
        try self.instructions.append(Instruction.init(tag, v));
    }
};

const Program = struct {
    const Blocks = std.SegmentedList(Block, 0);
    blocks: Blocks,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .blocks = Blocks{}, .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        self.blocks.deinit(self.allocator);
    }

    /// Block must be freed by the owner, before Program.deinit is called
    pub fn make_block(self: *@This()) !*Block {
        const block = try self.blocks.addOne(self.allocator);
        block.* = Block.init(self.allocator);
        return block;
    }

    pub fn dump(self: *@This(), writer: anytype) !void {
        var iter = self.blocks.iterator(0);
        var index: usize = 0;
        while (iter.next()) |block| : (index += 1) {
            try writer.print("Block {d}:\n", .{index + 1});
            try block.dump(writer, 4);
        }
    }
};

const UseGPA = builtin.mode == .Debug;

pub fn main() !void {
    var gpa = if (UseGPA) std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = if (UseGPA) gpa.allocator() else std.heap.c_allocator;
    defer _ = if (UseGPA) gpa.deinit();

    var program = Program.init(allocator);
    defer program.deinit();

    const block1 = try program.make_block();
    defer block1.deinit();

    const block2 = try program.make_block();
    defer block2.deinit();

    const block3 = try program.make_block();
    defer block3.deinit();

    const block4 = try program.make_block();
    defer block4.deinit();

    const block5 = try program.make_block();
    defer block5.deinit();

    const block6 = try program.make_block();
    defer block6.deinit();

    try block1.appendInit(.Store, 5);
    try block1.appendInit(.LoadImmediate, 0);
    try block1.appendInit(.SetLocal, 0);
    try block1.appendInit(.Load, 5);
    try block1.appendInit(.LoadImmediate, 0);
    try block1.appendInit(.Store, 6);
    try block1.appendInit(.Jump, block4);

    try block2.appendInit(.Exit, void{});

    try block3.appendInit(.LoadImmediate, 0);
    try block3.appendInit(.Jump, block5);

    try block4.appendInit(.GetLocal, 0);
    try block4.appendInit(.Store, 7);
    try block4.appendInit(.LoadImmediate, 100_000_000);
    try block4.appendInit(.LessThan, .{ .lhs = 7 });
    try block4.appendInit(.JumpConditional, .{
        .true_target = block3,
        .false_target = block6,
    });

    try block5.appendInit(.Store, 6);
    try block5.appendInit(.GetLocal, 0);
    try block5.appendInit(.Increment, void{});
    try block5.appendInit(.SetLocal, 0);
    try block5.appendInit(.Jump, block4);

    try block6.appendInit(.Load, 6);
    try block6.appendInit(.Jump, block2);

    try program.dump(std.io.getStdOut().writer());

    var vm = try VM.init(allocator);
    defer vm.deinit();

    var jit = JIT.init(allocator);
    defer jit.deinit();

    var executable = try jit.compile(&program);
    try executable.run(&vm);

    try vm.dump(std.io.getStdOut().writer());
}

const Executable = struct {
    code: []u8,
    code_size: usize,

    pub const JITFunction = *const fn (usize, []Value, locals: []Value) void;

    pub fn run(self: *@This(), vm: *VM) !void {
        // RDI: VM&
        // RSI: Value* registers
        // RDX: Value* locals
        // const func: JITFunction = @ptrCast(@alignCast(self.code));
        //  func(0, vm.registers.items.ptr, vm.locals.items.ptr);

        asm volatile ("call *%%rcx"
            :
            : [unholyRegion] "{rcx}" (self.code.ptr),
              [a] "{rsi}" (vm.registers.items.ptr),
              [b] "{rdx}" (vm.locals.items.ptr),
        );
    }
};

const Assembler = struct {
    output: std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{
            .output = std.ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.output.deinit();
    }

    const RegType = u8;
    pub const Reg = enum(RegType) {
        GPR0 = 0, // RAX
        GPR1 = 1, // RCX

        RegisterArrayBase = 6, // RSI
        LocalsArrayBase = 2, // RDX

        pub fn to_underlying(self: @This()) RegType {
            return @intFromEnum(self);
        }
    };

    const Operand = union(enum) {
        reg: Reg,
        imm64: Value,
        mem_64_base_and_offset: struct {
            base: Reg,
            offset: u32,
        },

        pub const init = UnionInit(@This());
    };

    const deadbeef = 0xdeadbeef;

    pub fn mov(self: *@This(), dst: Operand, src: Operand) !void {
        if (dst == .reg and src == .reg) {
            try self.emit8(0x48);
            try self.emit8(0x89);
            try self.emit8(0xc0 | (dst.reg.to_underlying() << 3) | src.reg.to_underlying());
            return;
        }

        if (dst == .reg and src == .imm64) {
            try self.emit8(0x48);
            try self.emit8(0xb8 | dst.reg.to_underlying());
            try self.emit64(src.imm64);
            return;
        }

        if (dst == .mem_64_base_and_offset and src == .reg) {
            try self.emit8(0x48);
            try self.emit8(0x89);
            try self.emit8(0x80 | (src.reg.to_underlying() << 3) | dst.mem_64_base_and_offset.base.to_underlying());
            try self.emit32(dst.mem_64_base_and_offset.offset);
            return;
        }

        if (dst == .reg and src == .mem_64_base_and_offset) {
            // MOV dst, QWORD [src.base + src.offset]
            try self.emit8(0x48);
            try self.emit8(0x8b);
            try self.emit8(0x80 | (dst.reg.to_underlying() << 3) | src.mem_64_base_and_offset.base.to_underlying());
            try self.emit32(src.mem_64_base_and_offset.offset);
            return;
        }

        unreachable;
    }

    pub fn emit8(self: *@This(), val: u8) !void {
        try self.output.append(val);
    }

    pub fn emit32(self: *@This(), val: u32) !void {
        try self.output.writer().writeInt(u32, val, .little);
    }

    pub fn emit64(self: *@This(), val: u64) !void {
        try self.output.writer().writeInt(u64, val, .little);
    }

    pub fn emit(self: *@This(), comptime T: type, val: T) !void {
        try self.output.writer().writeInt(T, val, .little);
    }

    // Instructions
    pub fn load_immediate64(self: *@This(), dst: Reg, imm: Value) !void {
        try self.mov(Operand.init(.reg, dst), Operand.init(.imm64, imm));
    }

    pub fn store_vm_register(self: *@This(), dst: VMRegister, src: Reg) !void {
        try self.mov(Operand.init(.mem_64_base_and_offset, .{
            .base = Reg.RegisterArrayBase,
            .offset = @as(u32, @intCast(dst * @sizeOf(Value))),
        }), Operand.init(.reg, src));
    }

    pub fn load_vm_register(self: *@This(), dst: Reg, src: VMRegister) !void {
        try self.mov(Operand.init(.reg, dst), Operand.init(.mem_64_base_and_offset, .{
            .base = Reg.RegisterArrayBase,
            .offset = @as(u32, @intCast(src * @sizeOf(Value))),
        }));
    }

    pub fn store_vm_local(self: *@This(), dst: VMLocal, src: Reg) !void {
        try self.mov(Operand.init(.mem_64_base_and_offset, .{
            .base = Reg.LocalsArrayBase,
            .offset = @as(u32, @intCast(dst * @sizeOf(Value))),
        }), Operand.init(.reg, src));
    }

    pub fn load_vm_local(self: *@This(), dst: Reg, src: VMLocal) !void {
        try self.mov(Operand.init(.reg, dst), Operand.init(.mem_64_base_and_offset, .{
            .base = Reg.LocalsArrayBase,
            .offset = @as(u32, @intCast(src * @sizeOf(Value))),
        }));
    }

    pub fn increment(self: *@This(), dst: Reg) !void {
        try self.emit8(0x48);
        try self.emit8(0xff);
        try self.emit8(0xc0 | dst.to_underlying());
    }

    pub fn less_than(self: *@This(), dst: Reg, src: Reg) !void {
        // cmp src, dst
        try self.emit8(0x48);
        try self.emit8(0x39);
        try self.emit8(0xc0 | (src.to_underlying() << 3) | dst.to_underlying());

        // setl dst
        try self.emit8(0x0f);
        try self.emit8(0x9c);
        try self.emit8(0xc0 | dst.to_underlying());

        // movzx dst, dst
        try self.emit8(0x48);
        try self.emit8(0x0f);
        try self.emit8(0xb6);
        try self.emit8(0xc0 | (dst.to_underlying() << 3) | dst.to_underlying());
    }

    pub fn jump(self: *@This(), target: *Block) !void {
        // jmp target (RIP-relative 32-bit offset)
        try self.emit8(0xe9);
        try target.jumps.append(self.output.items.len);
        try self.emit32(deadbeef);
    }

    pub fn jump_conditional(self: *@This(), reg: Reg, true_target: *Block, false_target: *Block) !void {
        // if reg == 0 jump to false_false else jump to true_target
        // cmp reg, 0
        try self.emit8(0x48);
        try self.emit8(0x83);
        try self.emit8(0xf8);
        try self.emit8(0x00 | reg.to_underlying());

        // jz false_target (RIP-relative 32-bit offset)
        try self.emit8(0x0f);
        try self.emit8(0x84);
        try false_target.jumps.append(self.output.items.len);
        try self.emit32(deadbeef);

        // jmp true_target (RIP-relative 32-bit offset)
        try self.jump(true_target);
    }

    pub fn exit(self: *@This()) !void {
        try self.emit8(0xc3);
    }
};

const JIT = struct {
    assembler: Assembler,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{
            .allocator = allocator,
            .assembler = Assembler.init(allocator),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.assembler.deinit();
    }

    pub fn compile(self: *@This(), program: *Program) !Executable {
        var iter = program.blocks.iterator(0);

        while (iter.next()) |block| {
            block.offset = self.assembler.output.items.len;

            for (block.instructions.items) |instruction| {
                switch (instruction) {
                    .LoadImmediate => |li| try self.compile_load_immediate(li),
                    .Load => |l| try self.compile_load(l),
                    .Store => |s| try self.compile_store(s),
                    .SetLocal => |sl| try self.compile_set_local(sl),
                    .GetLocal => |gl| try self.compile_get_local(gl),
                    .Increment => try self.compile_increment(),
                    .LessThan => |lt| try self.compile_less_than(lt),
                    .Jump => |jmp| try self.compile_jump(jmp),
                    .JumpConditional => |jmpc| try self.compile_jump_conditional(jmpc),
                    .Exit => try self.compile_exit(),
                }
            }
        }

        iter = program.blocks.iterator(0);

        while (iter.next()) |b| {
            for (b.jumps.items) |j| {
                const offset: i64 = @as(i64, @intCast(@as(i64, @intCast(b.offset)) - @as(i64, @intCast(j)) - @as(i64, @intCast(4))));

                self.assembler.output.items[j + 0] = @as(u8, @intCast((offset >> @intCast(0)) & 0xff));
                self.assembler.output.items[j + 1] = @as(u8, @intCast((offset >> @intCast(8)) & 0xff));
                self.assembler.output.items[j + 2] = @as(u8, @intCast((offset >> @intCast(16)) & 0xff));
                self.assembler.output.items[j + 3] = @as(u8, @intCast((offset >> @intCast(24)) & 0xff));
            }
        }

        try std.fs.cwd().writeFile("output.asm", self.assembler.output.items);

        // TODO: suckma
        const executable_memory = try std.os.mmap(null, std.mem.alignForward(usize, self.assembler.output.items.len, std.mem.page_size), std.os.PROT.READ | std.os.PROT.WRITE, std.os.MAP.ANONYMOUS | std.os.MAP.PRIVATE, 0, 0);
        @memcpy(executable_memory[0..self.assembler.output.items.len], self.assembler.output.items);

        try std.os.mprotect(executable_memory, std.os.PROT.READ | std.os.PROT.EXEC);

        return .{
            .code = executable_memory,
            .code_size = self.assembler.output.items.len,
        };
    }

    pub inline fn compile_load_immediate(self: *@This(), li: Value) !void {
        try self.assembler.load_immediate64(.GPR0, li);
        try self.assembler.store_vm_register(0, .GPR0);
    }

    pub inline fn compile_load(self: *@This(), l: VMRegister) !void {
        try self.assembler.load_vm_register(.GPR0, l);
        try self.assembler.store_vm_register(0, .GPR0);
    }

    pub inline fn compile_store(self: *@This(), s: usize) !void {
        try self.assembler.load_vm_register(.GPR0, 0);
        try self.assembler.store_vm_register(s, .GPR0);
    }

    pub inline fn compile_get_local(self: *@This(), gl: usize) !void {
        try self.assembler.load_vm_local(.GPR0, gl);
        try self.assembler.store_vm_register(0, .GPR0);
    }

    pub inline fn compile_set_local(self: *@This(), sl: usize) !void {
        try self.assembler.load_vm_register(.GPR0, 0);
        try self.assembler.store_vm_local(sl, .GPR0);
    }

    pub inline fn compile_increment(self: *@This()) !void {
        try self.assembler.load_vm_register(.GPR0, 0);
        try self.assembler.increment(.GPR0);
        try self.assembler.store_vm_register(0, .GPR0);
    }

    pub inline fn compile_less_than(self: *@This(), lt: LessThan) !void {
        try self.assembler.load_vm_register(.GPR0, lt.lhs);
        try self.assembler.load_vm_register(.GPR1, 0);
        try self.assembler.less_than(.GPR0, .GPR1);
        try self.assembler.store_vm_register(0, .GPR0);
    }

    pub inline fn compile_jump(self: *@This(), jmp: *Block) !void {
        try self.assembler.jump(jmp);
    }

    pub inline fn compile_jump_conditional(self: *@This(), jmpc: JumpConditional) !void {
        try self.assembler.load_vm_register(.GPR0, 0);
        try self.assembler.jump_conditional(.GPR0, jmpc.true_target, jmpc.false_target);
    }

    pub inline fn compile_exit(self: *@This()) !void {
        try self.assembler.exit();
    }
};
