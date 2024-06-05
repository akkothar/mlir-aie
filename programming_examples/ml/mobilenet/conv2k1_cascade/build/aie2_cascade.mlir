module {
  aie.device(npu1_1col) {
    func.func private @conv2dk1_skip_ui8_i8_i8_get(memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32)
    func.func private @conv2dk1_ui8_put(memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_5 = aie.tile(0, 5)
    %tile_0_4 = aie.tile(0, 4)
    aie.cascade_flow(%tile_0_5, %tile_0_4)
    aie.objectfifo @inOF_act_L3L2(%tile_0_0, {%tile_0_5, %tile_0_4}, [2 : i32, 2 : i32, 2 : i32]) : !aie.objectfifo<memref<7x1x64xui8>>
    aie.objectfifo @OF_wts_L3L2(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @OF_wts_memtile_put(%tile_0_1, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @OF_wts_memtile_get(%tile_0_1, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo.link [@OF_wts_L3L2] -> [@OF_wts_memtile_put, @OF_wts_memtile_get]()
    %rtp04 = aie.buffer(%tile_0_4) {sym_name = "rtp04"} : memref<16xi32> 
    aie.objectfifo @OF_bn_12(%tile_0_4, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<7x1x64xi8>>
    aie.objectfifo @outOFL2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<7x1x64xi8>>
    aie.objectfifo.link [@OF_bn_12] -> [@outOFL2L3]()
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @OF_wts_memtile_put(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
        %c0_0 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c7 step %c1_1 {
          %2 = aie.objectfifo.acquire @inOF_act_L3L2(Consume, 1) : !aie.objectfifosubview<memref<7x1x64xui8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<7x1x64xui8>> -> memref<7x1x64xui8>
          %c7_i32 = arith.constant 7 : i32
          %c64_i32 = arith.constant 64 : i32
          %c64_i32_2 = arith.constant 64 : i32
          func.call @conv2dk1_ui8_put(%3, %1, %c7_i32, %c64_i32, %c64_i32_2) : (memref<7x1x64xui8>, memref<2048xi8>, i32, i32, i32) -> ()
          aie.objectfifo.release @inOF_act_L3L2(Consume, 1)
        }
        aie.objectfifo.release @OF_wts_memtile_put(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_skip_put.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @OF_wts_memtile_get(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtp04[%c0_0] : memref<16xi32>
        %c1_1 = arith.constant 1 : index
        %3 = memref.load %rtp04[%c1_1] : memref<16xi32>
        %c0_2 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c7 step %c1_3 {
          %4 = aie.objectfifo.acquire @inOF_act_L3L2(Consume, 1) : !aie.objectfifosubview<memref<7x1x64xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<7x1x64xui8>> -> memref<7x1x64xui8>
          %6 = aie.objectfifo.acquire @OF_bn_12(Produce, 1) : !aie.objectfifosubview<memref<7x1x64xi8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<7x1x64xi8>> -> memref<7x1x64xi8>
          %c7_i32 = arith.constant 7 : i32
          %c64_i32 = arith.constant 64 : i32
          %c64_i32_4 = arith.constant 64 : i32
          func.call @conv2dk1_skip_ui8_i8_i8_get(%5, %1, %7, %5, %c7_i32, %c64_i32, %c64_i32_4, %2, %3) : (memref<7x1x64xui8>, memref<2048xi8>, memref<7x1x64xi8>, memref<7x1x64xui8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @inOF_act_L3L2(Consume, 1)
          aie.objectfifo.release @OF_bn_12(Produce, 1)
        }
        aie.objectfifo.release @OF_wts_memtile_get(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_skip_get.o"}
    func.func @sequence(%arg0: memref<784xi32>, %arg1: memref<1024xi32>, %arg2: memref<784xi32>) {
      aiex.npu.rtp_write(0, 4, 0, 11) {buffer_sym_name = "rtp04"}
      aiex.npu.rtp_write(0, 4, 1, 0) {buffer_sym_name = "rtp04"}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 784][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<784xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 784][0, 0, 0]) {id = 2 : i64, metadata = @outOFL2L3} : memref<784xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @OF_wts_L3L2} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

