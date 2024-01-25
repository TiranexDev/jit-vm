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

const Instruction = union(enum) {
    LoadImmediate: Value,
    Load: VMRegister,
    Store: VMRegister,
    SetLocal: VMLocal,
    GetLocal: VMLocal,
    Increment: void,
    LessThan: struct { lhs: VMRegister }, // lhs
    Jump: *Block,
    JumpConditional: struct {
        true_target: *Block,
        false_target: *Block,
    },

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
            .locals = try Locals.initCapacity(allocator, 24),
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
};

const Block = struct {
    const Instructions = std.ArrayList(Instruction);
    instructions: Instructions,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{
            .instructions = Instructions.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.instructions.deinit();
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

    var current_block = block1;
    var ic: usize = 0; // Instruction Counter

    var vm = try VM.init(allocator);
    defer vm.deinit();

    o: while (true) {
        if (ic >= current_block.instructions.items.len) {
            break :o;
        }

        const instruction = current_block.instructions.items[ic];

        switch (instruction) {
            .LoadImmediate => |li| {
                vm.registers.items[0] = li;
            },
            .Load => |l| {
                vm.registers.items[0] = vm.registers.items[l];
            },
            .Store => |s| {
                vm.registers.items[s] = vm.registers.items[0];
            },
            .SetLocal => |sl| vm.locals.items[sl] = vm.registers.items[0], //try vm.setLocal(sl, vm.registers.items[0]),
            .GetLocal => |gl| vm.registers.items[0] = vm.locals.items[gl],
            .Increment => {
                vm.registers.items[0] += 1;
            },
            .LessThan => |lt| {
                vm.registers.items[0] = if (vm.registers.items[lt.lhs] < vm.registers.items[0]) 1 else 0;
            },
            .Jump => |jmp| {
                current_block = jmp;
                ic = 0;
                continue :o;
            },
            .JumpConditional => |jmpc| {
                current_block = if (vm.registers.items[0] == 1)
                    jmpc.true_target
                else
                    jmpc.false_target;
                ic = 0;
                continue :o;
            },
        }

        ic += 1;
    }
}
