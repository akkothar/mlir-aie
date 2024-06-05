module attributes {llvm.target_triple = "aie2"} {
  llvm.mlir.global external @rtp04() {addr_space = 0 : i32} : !llvm.array<16 x i32>
  llvm.mlir.global external @inOF_act_L3L2_1_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_1_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_0_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_0_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @OF_wts_L3L2_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.mlir.global external @OF_wts_memtile_put_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @OF_wts_memtile_get_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @OF_bn_12_buff_1() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @OF_bn_12_buff_0() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @OF_bn_12_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @OF_bn_12_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.mlir.global external @outOFL2L3_cons() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @outOFL2L3() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @OF_bn_12_cons() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @OF_bn_12() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @OF_wts_memtile_get_cons() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @OF_wts_memtile_get() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @OF_wts_memtile_put_cons() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @OF_wts_memtile_put() {addr_space = 0 : i32} : !llvm.array<2048 x i8>
  llvm.mlir.global external @OF_wts_L3L2_cons() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.mlir.global external @OF_wts_L3L2() {addr_space = 0 : i32} : !llvm.array<4096 x i8>
  llvm.mlir.global external @inOF_act_L3L2_0_cons() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2_1_cons() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.mlir.global external @inOF_act_L3L2() {addr_space = 0 : i32} : !llvm.array<7 x array<1 x array<64 x i8>>>
  llvm.func @conv2dk1_skip_ui8_i8_i8_get(!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @conv2dk1_ui8_put(!llvm.ptr, !llvm.ptr, i32, i32, i32) attributes {sym_visibility = "private"}
  llvm.func @sequence(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
    llvm.return
  }
  llvm.func @core_0_4() {
    %0 = llvm.mlir.addressof @inOF_act_L3L2_1_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @OF_bn_12_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @inOF_act_L3L2_1_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @OF_wts_memtile_get_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.addressof @OF_bn_12_buff_0 : !llvm.ptr
    %5 = llvm.mlir.constant(31 : index) : i64
    %6 = llvm.mlir.addressof @rtp04 : !llvm.ptr
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(4294967294 : index) : i64
    %9 = llvm.mlir.constant(50 : i32) : i32
    %10 = llvm.mlir.constant(53 : i32) : i32
    %11 = llvm.mlir.constant(48 : i32) : i32
    %12 = llvm.mlir.constant(52 : i32) : i32
    %13 = llvm.mlir.constant(49 : i32) : i32
    %14 = llvm.mlir.constant(51 : i32) : i32
    %15 = llvm.mlir.constant(1 : i32) : i32
    %16 = llvm.mlir.constant(64 : i32) : i32
    %17 = llvm.mlir.constant(7 : i32) : i32
    %18 = llvm.mlir.constant(6 : index) : i64
    %19 = llvm.mlir.constant(-1 : i32) : i32
    %20 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%20 : i64)
  ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb8
    %22 = llvm.icmp "slt" %21, %8 : i64
    llvm.cond_br %22, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %19) : (i32, i32) -> ()
    %23 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %24 = llvm.ptrtoint %23 : !llvm.ptr to i64
    %25 = llvm.and %24, %5  : i64
    %26 = llvm.icmp "eq" %25, %20 : i64
    "llvm.intr.assume"(%26) : (i1) -> ()
    %27 = llvm.load %23 : !llvm.ptr -> i32
    "llvm.intr.assume"(%26) : (i1) -> ()
    %28 = llvm.getelementptr %23[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %29 = llvm.load %28 : !llvm.ptr -> i32
    llvm.br ^bb3(%20 : i64)
  ^bb3(%30: i64):  // 2 preds: ^bb2, ^bb4
    %31 = llvm.icmp "slt" %30, %18 : i64
    llvm.cond_br %31, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %32 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.and %33, %5  : i64
    %35 = llvm.icmp "eq" %34, %20 : i64
    "llvm.intr.assume"(%35) : (i1) -> ()
    %36 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.and %37, %5  : i64
    %39 = llvm.icmp "eq" %38, %20 : i64
    "llvm.intr.assume"(%39) : (i1) -> ()
    %40 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.and %41, %5  : i64
    %43 = llvm.icmp "eq" %42, %20 : i64
    "llvm.intr.assume"(%43) : (i1) -> ()
    "llvm.intr.assume"(%43) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%40, %36, %32, %40, %17, %16, %16, %27, %29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %44 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %45 = llvm.ptrtoint %44 : !llvm.ptr to i64
    %46 = llvm.and %45, %5  : i64
    %47 = llvm.icmp "eq" %46, %20 : i64
    "llvm.intr.assume"(%47) : (i1) -> ()
    "llvm.intr.assume"(%39) : (i1) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %5  : i64
    %51 = llvm.icmp "eq" %50, %20 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%48, %36, %44, %48, %17, %16, %16, %27, %29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %52 = llvm.add %30, %7 : i64
    llvm.br ^bb3(%52 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %53 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %54 = llvm.ptrtoint %53 : !llvm.ptr to i64
    %55 = llvm.and %54, %5  : i64
    %56 = llvm.icmp "eq" %55, %20 : i64
    "llvm.intr.assume"(%56) : (i1) -> ()
    %57 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.and %58, %5  : i64
    %60 = llvm.icmp "eq" %59, %20 : i64
    "llvm.intr.assume"(%60) : (i1) -> ()
    %61 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %62 = llvm.ptrtoint %61 : !llvm.ptr to i64
    %63 = llvm.and %62, %5  : i64
    %64 = llvm.icmp "eq" %63, %20 : i64
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%61, %57, %53, %61, %17, %16, %16, %27, %29) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%14, %19) : (i32, i32) -> ()
    "llvm.intr.assume"(%26) : (i1) -> ()
    %65 = llvm.load %23 : !llvm.ptr -> i32
    "llvm.intr.assume"(%26) : (i1) -> ()
    %66 = llvm.load %28 : !llvm.ptr -> i32
    llvm.br ^bb6(%20 : i64)
  ^bb6(%67: i64):  // 2 preds: ^bb5, ^bb7
    %68 = llvm.icmp "slt" %67, %18 : i64
    llvm.cond_br %68, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %69 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %70 = llvm.ptrtoint %69 : !llvm.ptr to i64
    %71 = llvm.and %70, %5  : i64
    %72 = llvm.icmp "eq" %71, %20 : i64
    "llvm.intr.assume"(%72) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    %73 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %74 = llvm.ptrtoint %73 : !llvm.ptr to i64
    %75 = llvm.and %74, %5  : i64
    %76 = llvm.icmp "eq" %75, %20 : i64
    "llvm.intr.assume"(%76) : (i1) -> ()
    "llvm.intr.assume"(%76) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%73, %57, %69, %73, %17, %16, %16, %65, %66) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    "llvm.intr.assume"(%56) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    "llvm.intr.assume"(%64) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%61, %57, %53, %61, %17, %16, %16, %65, %66) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %77 = llvm.add %67, %7 : i64
    llvm.br ^bb6(%77 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %78 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %79 = llvm.ptrtoint %78 : !llvm.ptr to i64
    %80 = llvm.and %79, %5  : i64
    %81 = llvm.icmp "eq" %80, %20 : i64
    "llvm.intr.assume"(%81) : (i1) -> ()
    "llvm.intr.assume"(%60) : (i1) -> ()
    %82 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.and %83, %5  : i64
    %85 = llvm.icmp "eq" %84, %20 : i64
    "llvm.intr.assume"(%85) : (i1) -> ()
    "llvm.intr.assume"(%85) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%82, %57, %78, %82, %17, %16, %16, %65, %66) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    %86 = llvm.add %21, %7 : i64
    llvm.br ^bb1(%86 : i64)
  ^bb9:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%14, %19) : (i32, i32) -> ()
    %87 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %88 = llvm.ptrtoint %87 : !llvm.ptr to i64
    %89 = llvm.and %88, %5  : i64
    %90 = llvm.icmp "eq" %89, %20 : i64
    "llvm.intr.assume"(%90) : (i1) -> ()
    %91 = llvm.load %87 : !llvm.ptr -> i32
    "llvm.intr.assume"(%90) : (i1) -> ()
    %92 = llvm.getelementptr %87[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %93 = llvm.load %92 : !llvm.ptr -> i32
    llvm.br ^bb10(%20 : i64)
  ^bb10(%94: i64):  // 2 preds: ^bb9, ^bb11
    %95 = llvm.icmp "slt" %94, %18 : i64
    llvm.cond_br %95, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %96 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %97 = llvm.ptrtoint %96 : !llvm.ptr to i64
    %98 = llvm.and %97, %5  : i64
    %99 = llvm.icmp "eq" %98, %20 : i64
    "llvm.intr.assume"(%99) : (i1) -> ()
    %100 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %101 = llvm.ptrtoint %100 : !llvm.ptr to i64
    %102 = llvm.and %101, %5  : i64
    %103 = llvm.icmp "eq" %102, %20 : i64
    "llvm.intr.assume"(%103) : (i1) -> ()
    %104 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %105 = llvm.ptrtoint %104 : !llvm.ptr to i64
    %106 = llvm.and %105, %5  : i64
    %107 = llvm.icmp "eq" %106, %20 : i64
    "llvm.intr.assume"(%107) : (i1) -> ()
    "llvm.intr.assume"(%107) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%104, %100, %96, %104, %17, %16, %16, %91, %93) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %108 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %109 = llvm.ptrtoint %108 : !llvm.ptr to i64
    %110 = llvm.and %109, %5  : i64
    %111 = llvm.icmp "eq" %110, %20 : i64
    "llvm.intr.assume"(%111) : (i1) -> ()
    "llvm.intr.assume"(%103) : (i1) -> ()
    %112 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %113 = llvm.ptrtoint %112 : !llvm.ptr to i64
    %114 = llvm.and %113, %5  : i64
    %115 = llvm.icmp "eq" %114, %20 : i64
    "llvm.intr.assume"(%115) : (i1) -> ()
    "llvm.intr.assume"(%115) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%112, %100, %108, %112, %17, %16, %16, %91, %93) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    %116 = llvm.add %94, %7 : i64
    llvm.br ^bb10(%116 : i64)
  ^bb12:  // pred: ^bb10
    llvm.call @llvm.aie2.acquire(%13, %19) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%12, %19) : (i32, i32) -> ()
    %117 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %118 = llvm.ptrtoint %117 : !llvm.ptr to i64
    %119 = llvm.and %118, %5  : i64
    %120 = llvm.icmp "eq" %119, %20 : i64
    "llvm.intr.assume"(%120) : (i1) -> ()
    %121 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %122 = llvm.ptrtoint %121 : !llvm.ptr to i64
    %123 = llvm.and %122, %5  : i64
    %124 = llvm.icmp "eq" %123, %20 : i64
    "llvm.intr.assume"(%124) : (i1) -> ()
    %125 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %126 = llvm.ptrtoint %125 : !llvm.ptr to i64
    %127 = llvm.and %126, %5  : i64
    %128 = llvm.icmp "eq" %127, %20 : i64
    "llvm.intr.assume"(%128) : (i1) -> ()
    "llvm.intr.assume"(%128) : (i1) -> ()
    llvm.call @conv2dk1_skip_ui8_i8_i8_get(%125, %121, %117, %125, %17, %16, %16, %91, %93) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%11, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%10, %15) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%9, %15) : (i32, i32) -> ()
    llvm.return
  }
  llvm.func @core_0_5() {
    %0 = llvm.mlir.addressof @inOF_act_L3L2_0_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @inOF_act_L3L2_0_cons_buff_0 : !llvm.ptr
    %2 = llvm.mlir.constant(31 : index) : i64
    %3 = llvm.mlir.addressof @OF_wts_memtile_put_cons_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(4294967294 : index) : i64
    %5 = llvm.mlir.constant(50 : i32) : i32
    %6 = llvm.mlir.constant(48 : i32) : i32
    %7 = llvm.mlir.constant(49 : i32) : i32
    %8 = llvm.mlir.constant(51 : i32) : i32
    %9 = llvm.mlir.constant(1 : i32) : i32
    %10 = llvm.mlir.constant(64 : i32) : i32
    %11 = llvm.mlir.constant(7 : i32) : i32
    %12 = llvm.mlir.constant(6 : index) : i64
    %13 = llvm.mlir.constant(-1 : i32) : i32
    %14 = llvm.mlir.constant(2 : index) : i64
    %15 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%15 : i64)
  ^bb1(%16: i64):  // 2 preds: ^bb0, ^bb8
    %17 = llvm.icmp "slt" %16, %4 : i64
    llvm.cond_br %17, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    llvm.br ^bb3(%15 : i64)
  ^bb3(%18: i64):  // 2 preds: ^bb2, ^bb4
    %19 = llvm.icmp "slt" %18, %12 : i64
    llvm.cond_br %19, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    %20 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %21 = llvm.ptrtoint %20 : !llvm.ptr to i64
    %22 = llvm.and %21, %2  : i64
    %23 = llvm.icmp "eq" %22, %15 : i64
    "llvm.intr.assume"(%23) : (i1) -> ()
    %24 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %25 = llvm.ptrtoint %24 : !llvm.ptr to i64
    %26 = llvm.and %25, %2  : i64
    %27 = llvm.icmp "eq" %26, %15 : i64
    "llvm.intr.assume"(%27) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%24, %20, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    "llvm.intr.assume"(%23) : (i1) -> ()
    %28 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %29 = llvm.ptrtoint %28 : !llvm.ptr to i64
    %30 = llvm.and %29, %2  : i64
    %31 = llvm.icmp "eq" %30, %15 : i64
    "llvm.intr.assume"(%31) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%28, %20, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    %32 = llvm.add %18, %14 : i64
    llvm.br ^bb3(%32 : i64)
  ^bb5:  // pred: ^bb3
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    %33 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %34 = llvm.ptrtoint %33 : !llvm.ptr to i64
    %35 = llvm.and %34, %2  : i64
    %36 = llvm.icmp "eq" %35, %15 : i64
    "llvm.intr.assume"(%36) : (i1) -> ()
    %37 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.and %38, %2  : i64
    %40 = llvm.icmp "eq" %39, %15 : i64
    "llvm.intr.assume"(%40) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%37, %33, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%5, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    llvm.br ^bb6(%15 : i64)
  ^bb6(%41: i64):  // 2 preds: ^bb5, ^bb7
    %42 = llvm.icmp "slt" %41, %12 : i64
    llvm.cond_br %42, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %43 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.and %44, %2  : i64
    %46 = llvm.icmp "eq" %45, %15 : i64
    "llvm.intr.assume"(%46) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%43, %33, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    "llvm.intr.assume"(%40) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%37, %33, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    %47 = llvm.add %41, %14 : i64
    llvm.br ^bb6(%47 : i64)
  ^bb8:  // pred: ^bb6
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    "llvm.intr.assume"(%36) : (i1) -> ()
    %48 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.and %49, %2  : i64
    %51 = llvm.icmp "eq" %50, %15 : i64
    "llvm.intr.assume"(%51) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%48, %33, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%5, %9) : (i32, i32) -> ()
    %52 = llvm.add %16, %14 : i64
    llvm.br ^bb1(%52 : i64)
  ^bb9:  // pred: ^bb1
    llvm.call @llvm.aie2.acquire(%8, %13) : (i32, i32) -> ()
    llvm.br ^bb10(%15 : i64)
  ^bb10(%53: i64):  // 2 preds: ^bb9, ^bb11
    %54 = llvm.icmp "slt" %53, %12 : i64
    llvm.cond_br %54, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    %55 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.and %56, %2  : i64
    %58 = llvm.icmp "eq" %57, %15 : i64
    "llvm.intr.assume"(%58) : (i1) -> ()
    %59 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %60 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %61 = llvm.and %60, %2  : i64
    %62 = llvm.icmp "eq" %61, %15 : i64
    "llvm.intr.assume"(%62) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%59, %55, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    "llvm.intr.assume"(%58) : (i1) -> ()
    %63 = llvm.getelementptr %0[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.and %64, %2  : i64
    %66 = llvm.icmp "eq" %65, %15 : i64
    "llvm.intr.assume"(%66) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%63, %55, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    %67 = llvm.add %53, %14 : i64
    llvm.br ^bb10(%67 : i64)
  ^bb12:  // pred: ^bb10
    llvm.call @llvm.aie2.acquire(%7, %13) : (i32, i32) -> ()
    %68 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2048 x i8>
    %69 = llvm.ptrtoint %68 : !llvm.ptr to i64
    %70 = llvm.and %69, %2  : i64
    %71 = llvm.icmp "eq" %70, %15 : i64
    "llvm.intr.assume"(%71) : (i1) -> ()
    %72 = llvm.getelementptr %1[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<7 x array<1 x array<64 x i8>>>
    %73 = llvm.ptrtoint %72 : !llvm.ptr to i64
    %74 = llvm.and %73, %2  : i64
    %75 = llvm.icmp "eq" %74, %15 : i64
    "llvm.intr.assume"(%75) : (i1) -> ()
    llvm.call @conv2dk1_ui8_put(%72, %68, %11, %10, %10) : (!llvm.ptr, !llvm.ptr, i32, i32, i32) -> ()
    llvm.call @llvm.aie2.release(%6, %9) : (i32, i32) -> ()
    llvm.call @llvm.aie2.release(%5, %9) : (i32, i32) -> ()
    llvm.return
  }
}

