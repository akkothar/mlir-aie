module {
  aie.device(npu1_1col) {
    memref.global "public" @outOFL2L3_cons : memref<1x1x256xui8>
    memref.global "public" @outOFL2L3 : memref<1x1x256xui8>
    memref.global "public" @out_02_L2_cons : memref<1x1x256xui8>
    memref.global "public" @out_02_L2 : memref<1x1x256xui8>
    memref.global "public" @act_L2_02_cons : memref<7x7x256xui8>
    memref.global "public" @act_L2_02 : memref<7x7x256xui8>
    memref.global "public" @inOF_act_L3L2_cons : memref<7x7x256xui8>
    memref.global "public" @inOF_act_L3L2 : memref<7x7x256xui8>
    func.func private @average_pooling(memref<7x7x256xui8>, memref<1x1x256xui8>, i32, i32, i32, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %outOFL2L3_cons_prod_lock = aie.lock(%tile_0_0, 2) {init = 1 : i32, sym_name = "outOFL2L3_cons_prod_lock"}
    %outOFL2L3_cons_cons_lock = aie.lock(%tile_0_0, 3) {init = 0 : i32, sym_name = "outOFL2L3_cons_cons_lock"}
    %out_02_L2_cons_buff_0 = aie.buffer(%tile_0_1) {address = 12544 : i32, mem_bank = 0 : i32, sym_name = "out_02_L2_cons_buff_0"} : memref<1x1x256xui8> 
    %out_02_L2_cons_prod_lock = aie.lock(%tile_0_1, 2) {init = 1 : i32, sym_name = "out_02_L2_cons_prod_lock"}
    %out_02_L2_cons_cons_lock = aie.lock(%tile_0_1, 3) {init = 0 : i32, sym_name = "out_02_L2_cons_cons_lock"}
    %out_02_L2_buff_0 = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "out_02_L2_buff_0"} : memref<1x1x256xui8> 
    %out_02_L2_prod_lock = aie.lock(%tile_0_2, 2) {init = 1 : i32, sym_name = "out_02_L2_prod_lock"}
    %out_02_L2_cons_lock = aie.lock(%tile_0_2, 3) {init = 0 : i32, sym_name = "out_02_L2_cons_lock"}
    %act_L2_02_cons_buff_0 = aie.buffer(%tile_0_2) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "act_L2_02_cons_buff_0"} : memref<7x7x256xui8> 
    %act_L2_02_cons_prod_lock = aie.lock(%tile_0_2, 0) {init = 1 : i32, sym_name = "act_L2_02_cons_prod_lock"}
    %act_L2_02_cons_cons_lock = aie.lock(%tile_0_2, 1) {init = 0 : i32, sym_name = "act_L2_02_cons_cons_lock"}
    %inOF_act_L3L2_cons_buff_0 = aie.buffer(%tile_0_1) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "inOF_act_L3L2_cons_buff_0"} : memref<7x7x256xui8> 
    %inOF_act_L3L2_cons_prod_lock = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = "inOF_act_L3L2_cons_prod_lock"}
    %inOF_act_L3L2_cons_cons_lock = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_cons_cons_lock"}
    %inOF_act_L3L2_prod_lock = aie.lock(%tile_0_0, 0) {init = 1 : i32, sym_name = "inOF_act_L3L2_prod_lock"}
    %inOF_act_L3L2_cons_lock = aie.lock(%tile_0_0, 1) {init = 0 : i32, sym_name = "inOF_act_L3L2_cons_lock"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      cf.br ^bb1(%c0 : index)
    ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
      %1 = arith.cmpi slt, %0, %c9223372036854775807 : index
      cf.cond_br %1, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      aie.use_lock(%act_L2_02_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.use_lock(%out_02_L2_prod_lock, AcquireGreaterEqual, 1)
      %c7_i32 = arith.constant 7 : i32
      %c7_i32_1 = arith.constant 7 : i32
      %c256_i32 = arith.constant 256 : i32
      %c7_i32_2 = arith.constant 7 : i32
      func.call @average_pooling(%act_L2_02_cons_buff_0, %out_02_L2_buff_0, %c7_i32, %c7_i32_1, %c256_i32, %c7_i32_2) : (memref<7x7x256xui8>, memref<1x1x256xui8>, i32, i32, i32, i32) -> ()
      aie.use_lock(%act_L2_02_cons_prod_lock, Release, 1)
      aie.use_lock(%out_02_L2_cons_lock, Release, 1)
      %2 = arith.addi %0, %c1_0 : index
      cf.br ^bb1(%2 : index)
    ^bb3:  // pred: ^bb1
      aie.end
    } {link_with = "avg_pool.o"}
    aie.shim_dma_allocation @inOF_act_L3L2(MM2S, 0, 0)
    func.func @sequence(%arg0: memref<3136xi32>, %arg1: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 3136][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<3136xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 1 : i64, metadata = @outOFL2L3} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%inOF_act_L3L2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_cons_buff_0 : memref<7x7x256xui8>, 0, 12544) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%inOF_act_L3L2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%inOF_act_L3L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%inOF_act_L3L2_cons_buff_0 : memref<7x7x256xui8>, 0, 12544) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%inOF_act_L3L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      %2 = aie.dma_start(S2MM, 1, ^bb5, ^bb6)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%out_02_L2_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_02_L2_cons_buff_0 : memref<1x1x256xui8>, 0, 256) {bd_id = 24 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%out_02_L2_cons_cons_lock, Release, 1)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb4
      %3 = aie.dma_start(MM2S, 1, ^bb7, ^bb8)
    ^bb7:  // 2 preds: ^bb6, ^bb7
      aie.use_lock(%out_02_L2_cons_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_02_L2_cons_buff_0 : memref<1x1x256xui8>, 0, 256) {bd_id = 25 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%out_02_L2_cons_prod_lock, Release, 1)
      aie.next_bd ^bb7
    ^bb8:  // pred: ^bb6
      aie.end
    }
    aie.shim_dma_allocation @outOFL2L3(S2MM, 0, 0)
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%act_L2_02_cons_prod_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%act_L2_02_cons_buff_0 : memref<7x7x256xui8>, 0, 12544) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%act_L2_02_cons_cons_lock, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%out_02_L2_cons_lock, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_02_L2_buff_0 : memref<1x1x256xui8>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_02_L2_prod_lock, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb2
      aie.end
    }
  }
}
