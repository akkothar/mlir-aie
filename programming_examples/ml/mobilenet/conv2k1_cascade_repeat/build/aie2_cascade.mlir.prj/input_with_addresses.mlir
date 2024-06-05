module {
  aie.device(npu1_1col) {
    memref.global "public" @outOFL2L3_cons : memref<7x1x64xi8>
    memref.global "public" @outOFL2L3 : memref<7x1x64xi8>
    memref.global "public" @OF_bn_12_cons : memref<7x1x64xi8>
    memref.global "public" @OF_bn_12 : memref<7x1x64xi8>
    memref.global "public" @OF_wts_memtile_get_cons : memref<2048xi8>
    memref.global "public" @OF_wts_memtile_get : memref<2048xi8>
    memref.global "public" @OF_wts_memtile_put_cons : memref<2048xi8>
    memref.global "public" @OF_wts_memtile_put : memref<2048xi8>
    memref.global "public" @OF_wts_L3L2_cons : memref<4096xi8>
    memref.global "public" @OF_wts_L3L2 : memref<4096xi8>
    memref.global "public" @inOF_act_L3L2_0_cons : memref<7x1x64xui8>
    memref.global "public" @inOF_act_L3L2_1_cons : memref<7x1x64xui8>
    memref.global "public" @inOF_act_L3L2 : memref<7x1x64xui8>
    func.func private @conv2dk1_skip_ui8_i8_i8_get(memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32)
    func.func private @conv2dk1_skip_ui8_i8_put(memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_5 = aie.tile(0, 5)
    %tile_0_4 = aie.tile(0, 4)
    %outOFL2L3_cons_prod_lock = aie.lock(%tile_0_0, 4) {init = 0 : i32, sym_name = "outOFL2L3_cons_prod_lock"}
    %outOFL2L3_cons_cons_lock = aie.lock(%tile_0_0, 5) {init = 0 : i32, sym_name = "outOFL2L3_cons_cons_lock"}
    %OF_bn_12_cons_buff_0 = aie.buffer(%tile_0_1) {address = 4096 : i32, sym_name = "OF_bn_12_cons_buff_0"} : memref<7x1x64xi8> 
    %OF_bn_12_cons_buff_1 = aie.buffer(%tile_0_1) {address = 4544 : i32, sym_name = "OF_bn_12_cons_buff_1"} : memref<7x1x64xi8> 
    %OF_bn_12_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 2 : i32, sym_name = "OF_bn_12_cons_prod_lock"}
    %OF_bn_12_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "OF_bn_12_cons_cons_lock"}
    %OF_bn_12_buff_0 = aie.buffer(%tile_0_4) {address = 3072 : i32, sym_name = "OF_bn_12_buff_0"} : memref<7x1x64xi8> 
    %OF_bn_12_buff_1 = aie.buffer(%tile_0_4) {address = 3520 : i32, sym_name = "OF_bn_12_buff_1"} : memref<7x1x64xi8> 
    %OF_bn_12_prod_lock = aie.lock(%tile_0_4, 4) {init = 2 : i32, sym_name = "OF_bn_12_prod_lock"}
    %OF_bn_12_cons_lock = aie.lock(%tile_0_4, 5) {init = 0 : i32, sym_name = "OF_bn_12_cons_lock"}
    %OF_wts_memtile_get_cons_buff_0 = aie.buffer(%tile_0_4) {address = 1024 : i32, sym_name = "OF_wts_memtile_get_cons_buff_0"} : memref<2048xi8> 
    %OF_wts_memtile_get_cons_prod_lock = aie.lock(%tile_0_4, 2) {init = 1 : i32, sym_name = "OF_wts_memtile_get_cons_prod_lock"}
    %OF_wts_memtile_get_cons_cons_lock = aie.lock(%tile_0_4, 3) {init = 0 : i32, sym_name = "OF_wts_memtile_get_cons_cons_lock"}
    %OF_wts_memtile_put_cons_buff_0 = aie.buffer(%tile_0_5) {address = 1024 : i32, sym_name = "OF_wts_memtile_put_cons_buff_0"} : memref<2048xi8> 
    %OF_wts_memtile_put_cons_prod_lock = aie.lock(%tile_0_5, 2) {init = 1 : i32, sym_name = "OF_wts_memtile_put_cons_prod_lock"}
    %OF_wts_memtile_put_cons_cons_lock = aie.lock(%tile_0_5, 3) {init = 0 : i32, sym_name = "OF_wts_memtile_put_cons_cons_lock"}
    %OF_wts_L3L2_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "OF_wts_L3L2_cons_buff_0"} : memref<4096xi8> 
    %OF_wts_L3L2_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 2 : i32, sym_name = "OF_wts_L3L2_cons_prod_lock"}
    %OF_wts_L3L2_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "OF_wts_L3L2_cons_cons_lock"}
    %OF_wts_L3L2_prod_lock = aie.lock(%tile_0_0, 2) {init = 0 : i32, sym_name = "OF_wts_L3L2_prod_lock"}
    %OF_wts_L3L2_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "OF_wts_L3L2_cons_lock"}
    %inOF_act_L3L2_0_cons_buff_0 = aie.buffer(%tile_0_5) {address = 3072 : i32, sym_name = "inOF_act_L3L2_0_cons_buff_0"} : memref<7x1x64xui8> 
    %inOF_act_L3L2_0_cons_buff_1 = aie.buffer(%tile_0_5) {address = 3520 : i32, sym_name = "inOF_act_L3L2_0_cons_buff_1"} : memref<7x1x64xui8> 
    %inOF_act_L3L2_0_cons_prod_lock = aie.lock(%tile_0_5, 0) {init = 2 : i32, sym_name = "inOF_act_L3L2_0_cons_prod_lock"}
    %inOF_act_L3L2_0_cons_cons_lock = aie.lock(%tile_0_5, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_0_cons_cons_lock"}
    %inOF_act_L3L2_1_cons_buff_0 = aie.buffer(%tile_0_4) {address = 3968 : i32, sym_name = "inOF_act_L3L2_1_cons_buff_0"} : memref<7x1x64xui8> 
    %inOF_act_L3L2_1_cons_buff_1 = aie.buffer(%tile_0_4) {address = 4416 : i32, sym_name = "inOF_act_L3L2_1_cons_buff_1"} : memref<7x1x64xui8> 
    %inOF_act_L3L2_1_cons_prod_lock = aie.lock(%tile_0_4, 0) {init = 2 : i32, sym_name = "inOF_act_L3L2_1_cons_prod_lock"}
    %inOF_act_L3L2_1_cons_cons_lock = aie.lock(%tile_0_4, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_1_cons_cons_lock"}
    %inOF_act_L3L2_prod_lock = aie.lock(%tile_0_0, 0) {init = 0 : i32, sym_name = "inOF_act_L3L2_prod_lock"}
    %inOF_act_L3L2_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_0, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_5, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_4, DMA : 1)
    %rtp04 = aie.buffer(%tile_0_4) {address = 4864 : i32, sym_name = "rtp04"} : memref<16xi32> 
    aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_0, DMA : 0)
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      aie.use_lock(%OF_wts_memtile_put_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_0 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c1_1 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      %c2_2 = arith.constant 2 : index
      cf.br ^bb3(%c0_0 : index)
    ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
      %3 = arith.cmpi slt, %2, %c6 : index
      cf.cond_br %3, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32 = arith.constant 7 : i32
      %c64_i32 = arith.constant 64 : i32
      %c64_i32_3 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_0, %OF_wts_memtile_put_cons_buff_0, %c7_i32, %c64_i32, %c64_i32_3) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_4 = arith.constant 7 : i32
      %c64_i32_5 = arith.constant 64 : i32
      %c64_i32_6 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_1, %OF_wts_memtile_put_cons_buff_0, %c7_i32_4, %c64_i32_5, %c64_i32_6) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      %4 = arith.addi %2, %c2_2 : index
      cf.br ^bb3(%4 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_7 = arith.constant 7 : i32
      %c64_i32_8 = arith.constant 64 : i32
      %c64_i32_9 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_0, %OF_wts_memtile_put_cons_buff_0, %c7_i32_7, %c64_i32_8, %c64_i32_9) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_put_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_put_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_10 = arith.constant 0 : index
      %c7_11 = arith.constant 7 : index
      %c1_12 = arith.constant 1 : index
      %c6_13 = arith.constant 6 : index
      %c2_14 = arith.constant 2 : index
      cf.br ^bb6(%c0_10 : index)
    ^bb6(%5: index):  // 2 preds: ^bb5, ^bb7
      %6 = arith.cmpi slt, %5, %c6_13 : index
      cf.cond_br %6, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_15 = arith.constant 7 : i32
      %c64_i32_16 = arith.constant 64 : i32
      %c64_i32_17 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_1, %OF_wts_memtile_put_cons_buff_0, %c7_i32_15, %c64_i32_16, %c64_i32_17) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_18 = arith.constant 7 : i32
      %c64_i32_19 = arith.constant 64 : i32
      %c64_i32_20 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_0, %OF_wts_memtile_put_cons_buff_0, %c7_i32_18, %c64_i32_19, %c64_i32_20) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      %7 = arith.addi %5, %c2_14 : index
      cf.br ^bb6(%7 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_21 = arith.constant 7 : i32
      %c64_i32_22 = arith.constant 64 : i32
      %c64_i32_23 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_1, %OF_wts_memtile_put_cons_buff_0, %c7_i32_21, %c64_i32_22, %c64_i32_23) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_put_cons_prod_lock, Release, 1)
      %8 = arith.addi %0, %c2 : index
      cf.br ^bb1(%8 : index)
    ^bb9:  // pred: ^bb1
      aie.use_lock(%OF_wts_memtile_put_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_24 = arith.constant 0 : index
      %c7_25 = arith.constant 7 : index
      %c1_26 = arith.constant 1 : index
      %c6_27 = arith.constant 6 : index
      %c2_28 = arith.constant 2 : index
      cf.br ^bb10(%c0_24 : index)
    ^bb10(%9: index):  // 2 preds: ^bb9, ^bb11
      %10 = arith.cmpi slt, %9, %c6_27 : index
      cf.cond_br %10, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_29 = arith.constant 7 : i32
      %c64_i32_30 = arith.constant 64 : i32
      %c64_i32_31 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_0, %OF_wts_memtile_put_cons_buff_0, %c7_i32_29, %c64_i32_30, %c64_i32_31) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_32 = arith.constant 7 : i32
      %c64_i32_33 = arith.constant 64 : i32
      %c64_i32_34 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_1, %OF_wts_memtile_put_cons_buff_0, %c7_i32_32, %c64_i32_33, %c64_i32_34) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      %11 = arith.addi %9, %c2_28 : index
      cf.br ^bb10(%11 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, AcquireGreaterEqual, 1)
      %c7_i32_35 = arith.constant 7 : i32
      %c64_i32_36 = arith.constant 64 : i32
      %c64_i32_37 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_put(%inOF_act_L3L2_0_cons_buff_0, %OF_wts_memtile_put_cons_buff_0, %c7_i32_35, %c64_i32_36, %c64_i32_37) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_put_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "conv2dk1_skip_put.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c4294967294 = arith.constant 4294967294 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb8
      %1 = arith.cmpi slt, %0, %c4294967294 : index
      cf.cond_br %1, ^bb2, ^bb9
    ^bb2:  // pred: ^bb1
      aie.use_lock(%OF_wts_memtile_get_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_0 = arith.constant 0 : index
      %2 = memref.load %rtp04[%c0_0] : memref<16xi32>
      %c1_1 = arith.constant 1 : index
      %3 = memref.load %rtp04[%c1_1] : memref<16xi32>
      %c0_2 = arith.constant 0 : index
      %c7 = arith.constant 7 : index
      %c1_3 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      %c2_4 = arith.constant 2 : index
      cf.br ^bb3(%c0_2 : index)
    ^bb3(%4: index):  // 2 preds: ^bb2, ^bb4
      %5 = arith.cmpi slt, %4, %c6 : index
      cf.cond_br %5, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32 = arith.constant 7 : i32
      %c64_i32 = arith.constant 64 : i32
      %c64_i32_5 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_0, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_0, %inOF_act_L3L2_1_cons_buff_0, %c7_i32, %c64_i32, %c64_i32_5, %2, %3) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_6 = arith.constant 7 : i32
      %c64_i32_7 = arith.constant 64 : i32
      %c64_i32_8 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_1, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_1, %inOF_act_L3L2_1_cons_buff_1, %c7_i32_6, %c64_i32_7, %c64_i32_8, %2, %3) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      %6 = arith.addi %4, %c2_4 : index
      cf.br ^bb3(%6 : index)
    ^bb5:  // pred: ^bb3
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_9 = arith.constant 7 : i32
      %c64_i32_10 = arith.constant 64 : i32
      %c64_i32_11 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_0, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_0, %inOF_act_L3L2_1_cons_buff_0, %c7_i32_9, %c64_i32_10, %c64_i32_11, %2, %3) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_get_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_get_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_12 = arith.constant 0 : index
      %7 = memref.load %rtp04[%c0_12] : memref<16xi32>
      %c1_13 = arith.constant 1 : index
      %8 = memref.load %rtp04[%c1_13] : memref<16xi32>
      %c0_14 = arith.constant 0 : index
      %c7_15 = arith.constant 7 : index
      %c1_16 = arith.constant 1 : index
      %c6_17 = arith.constant 6 : index
      %c2_18 = arith.constant 2 : index
      cf.br ^bb6(%c0_14 : index)
    ^bb6(%9: index):  // 2 preds: ^bb5, ^bb7
      %10 = arith.cmpi slt, %9, %c6_17 : index
      cf.cond_br %10, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_19 = arith.constant 7 : i32
      %c64_i32_20 = arith.constant 64 : i32
      %c64_i32_21 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_1, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_1, %inOF_act_L3L2_1_cons_buff_1, %c7_i32_19, %c64_i32_20, %c64_i32_21, %7, %8) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_22 = arith.constant 7 : i32
      %c64_i32_23 = arith.constant 64 : i32
      %c64_i32_24 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_0, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_0, %inOF_act_L3L2_1_cons_buff_0, %c7_i32_22, %c64_i32_23, %c64_i32_24, %7, %8) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      %11 = arith.addi %9, %c2_18 : index
      cf.br ^bb6(%11 : index)
    ^bb8:  // pred: ^bb6
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_25 = arith.constant 7 : i32
      %c64_i32_26 = arith.constant 64 : i32
      %c64_i32_27 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_1, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_1, %inOF_act_L3L2_1_cons_buff_1, %c7_i32_25, %c64_i32_26, %c64_i32_27, %7, %8) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_get_cons_prod_lock, Release, 1)
      %12 = arith.addi %0, %c2 : index
      cf.br ^bb1(%12 : index)
    ^bb9:  // pred: ^bb1
      aie.use_lock(%OF_wts_memtile_get_cons_cons_lock, AcquireGreaterEqual, 1)
      %c0_28 = arith.constant 0 : index
      %13 = memref.load %rtp04[%c0_28] : memref<16xi32>
      %c1_29 = arith.constant 1 : index
      %14 = memref.load %rtp04[%c1_29] : memref<16xi32>
      %c0_30 = arith.constant 0 : index
      %c7_31 = arith.constant 7 : index
      %c1_32 = arith.constant 1 : index
      %c6_33 = arith.constant 6 : index
      %c2_34 = arith.constant 2 : index
      cf.br ^bb10(%c0_30 : index)
    ^bb10(%15: index):  // 2 preds: ^bb9, ^bb11
      %16 = arith.cmpi slt, %15, %c6_33 : index
      cf.cond_br %16, ^bb11, ^bb12
    ^bb11:  // pred: ^bb10
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_35 = arith.constant 7 : i32
      %c64_i32_36 = arith.constant 64 : i32
      %c64_i32_37 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_0, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_0, %inOF_act_L3L2_1_cons_buff_0, %c7_i32_35, %c64_i32_36, %c64_i32_37, %13, %14) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_38 = arith.constant 7 : i32
      %c64_i32_39 = arith.constant 64 : i32
      %c64_i32_40 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_1, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_1, %inOF_act_L3L2_1_cons_buff_1, %c7_i32_38, %c64_i32_39, %c64_i32_40, %13, %14) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      %17 = arith.addi %15, %c2_34 : index
      cf.br ^bb10(%17 : index)
    ^bb12:  // pred: ^bb10
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%OF_bn_12_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32_41 = arith.constant 7 : i32
      %c64_i32_42 = arith.constant 64 : i32
      %c64_i32_43 = arith.constant 64 : i32
      func.call @conv2dk1_skip_ui8_i8_i8_get(%inOF_act_L3L2_1_cons_buff_0, %OF_wts_memtile_get_cons_buff_0, %OF_bn_12_buff_0, %inOF_act_L3L2_1_cons_buff_0, %c7_i32_41, %c64_i32_42, %c64_i32_43, %13, %14) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, Release, 1)
      aie.use_lock(%OF_bn_12_cons_lock, Release, 1)
      aie.use_lock(%OF_wts_memtile_get_cons_prod_lock, Release, 1)
      aie.end
    } {link_with = "conv2dk1_skip_get.o"}
    aie.shim_dma_allocation @inOF_act_L3L2(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<784xi32>, %arg1: memref<1024xi32>, %arg2: memref<784xi32>) {
      aiex.npu.rtp_write(0, 4, 0, 11) {buffer_sym_name = "rtp04"}
      aiex.npu.rtp_write(0, 4, 1, 0) {buffer_sym_name = "rtp04"}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 784][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<784xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 784][0, 0, 0]) {id = 2 : i64, metadata = @outOFL2L3} : memref<784xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @OF_wts_L3L2} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_0_cons_buff_0 : memref<7x1x64xui8>, 0, 448) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inOF_act_L3L2_0_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_0_cons_buff_1 : memref<7x1x64xui8>, 0, 448) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inOF_act_L3L2_0_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%OF_wts_memtile_put_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_wts_memtile_put_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_wts_memtile_put_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      aie.end
    }
    aie.shim_dma_allocation @OF_wts_L3L2(MM2S, 1, 0)
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_0 : memref<7x1x64xui8>, 0, 448) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%inOF_act_L3L2_1_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_1_cons_buff_1 : memref<7x1x64xui8>, 0, 448) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inOF_act_L3L2_1_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%OF_wts_memtile_get_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_wts_memtile_get_cons_buff_0 : memref<2048xi8>, 0, 2048) {bd_id = 2 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_wts_memtile_get_cons_cons_lock, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
    ^bb6:  // 2 preds: ^bb5, ^bb7
      aie.use_lock(%OF_bn_12_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_bn_12_buff_0 : memref<7x1x64xi8>, 0, 448) {bd_id = 3 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%OF_bn_12_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb7:  // pred: ^bb6
      aie.use_lock(%OF_bn_12_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_bn_12_buff_1 : memref<7x1x64xi8>, 0, 448) {bd_id = 4 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_bn_12_prod_lock, Release, 1)
      aie.next_bd ^bb6
    ^bb8:  // pred: ^bb5
      aie.end
    }
    aie.shim_dma_allocation @outOFL2L3(S2MM, 0, 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%OF_wts_L3L2_cons_prod_lock, AcquireGreaterEqual, 2)
      aie.dma_bd(%OF_wts_L3L2_cons_buff_0 : memref<4096xi8>, 0, 4096) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%OF_wts_L3L2_cons_cons_lock, Release, 2)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%OF_wts_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_wts_L3L2_cons_buff_0 : memref<4096xi8>, 0, 2048) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%OF_wts_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(MM2S, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%OF_wts_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_wts_L3L2_cons_buff_0 : memref<4096xi8>, 2048, 2048) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%OF_wts_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(S2MM, 1, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%OF_bn_12_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_bn_12_cons_buff_0 : memref<7x1x64xi8>, 0, 448) {bd_id = 25 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%OF_bn_12_cons_cons_lock, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%OF_bn_12_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_bn_12_cons_buff_1 : memref<7x1x64xi8>, 0, 448) {bd_id = 26 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%OF_bn_12_cons_cons_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %4 = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%OF_bn_12_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_bn_12_cons_buff_0 : memref<7x1x64xi8>, 0, 448) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%OF_bn_12_cons_prod_lock, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%OF_bn_12_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%OF_bn_12_cons_buff_1 : memref<7x1x64xi8>, 0, 448) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%OF_bn_12_cons_prod_lock, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      aie.end
    }
    aie.configure_cascade(%tile_0_4, North, South)
    aie.configure_cascade(%tile_0_5, North, South)
  }
}
