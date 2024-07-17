module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @inOF_act_L3L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<7 x array<7 x array<256 x i8>>>
  llvm.mlir.global external @act_L2_02_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<7 x array<7 x array<256 x i8>>>
  llvm.mlir.global external @out_02_L2_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @out_02_L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<256 x i8>>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @outOFL2L3_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @outOFL2L3() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @out_02_L2_cons() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @out_02_L2() {addr_space = 0 : i32} : !llvm.array<1 x array<1 x array<256 x i8>>>
  llvm.mlir.global external @act_L2_02_cons() {addr_space = 0 : i32} : !llvm.array<7 x array<7 x array<256 x i8>>>
  llvm.mlir.global external @act_L2_02() {addr_space = 0 : i32} : !llvm.array<7 x array<7 x array<256 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_cons() {addr_space = 0 : i32} : !llvm.array<7 x array<7 x array<256 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2() {addr_space = 0 : i32} : !llvm.array<7 x array<7 x array<256 x i8>>>
  llvm.func @average_pooling(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @sequence(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    llvm.return
  }
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @act_L2_02_cons_buff_0 : !llvm.ptr
    %1 = llvm.mlir.constant(31 : index) : i64
    %2 = llvm.mlir.addressof @out_02_L2_buff_0 : !llvm.ptr
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(9223372036854775807 : index) : i64
    %5 = llvm.mlir.constant(51 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(50 : i32) : i32
    %8 = llvm.mlir.constant(49 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(256 : i32) : i32
    %11 = llvm.mlir.constant(7 : i32) : i32
    %12 = llvm.mlir.constant(-1 : i32) : i32
    %13 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%13 : i64)
  ^bb1(%14: i64):  // 2 preds: ^bb0, ^bb2
    %15 = llvm.icmp "slt" %14, %4 : i64
    llvm.cond_br %15, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%8, %12) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%7, %12) : (i32, i32) -> ()
    %16 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<1 x array<256 x i8>>>
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.and %17, %1  : i64
    %19 = llvm.icmp "eq" %18, %13 : i64
    "llvm.intr.assume"(%19) : (i1) -> ()
    %20 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<7 x array<256 x i8>>>
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.and %21, %1  : i64
    %23 = llvm.icmp "eq" %22, %13 : i64
    "llvm.intr.assume"(%23) : (i1) -> ()
    llvm.call @average_pooling(%20, %16, %11, %11, %10, %11) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%5, %9) : (i32, i32) -> ()
    %24 = llvm.add %14, %3 : i64
    llvm.br ^bb1(%24 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
}

