module {
  aie.device(npu1_1col) {
    func.func private @average_pooling(memref<7x7x256xui8>, memref<1x1x256xui8>, i32, i32, i32, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inOF_act_L3L2(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<7x7x256xui8>>
    aie.objectfifo @act_L2_02(%tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<7x7x256xui8>>
    aie.objectfifo.link [@inOF_act_L3L2] -> [@act_L2_02]([] [])
    aie.objectfifo @out_02_L2(%tile_0_2, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<1x1x256xui8>>
    aie.objectfifo @outOFL2L3(%tile_0_1, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<1x1x256xui8>>
    aie.objectfifo.link [@out_02_L2] -> [@outOFL2L3]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @act_L2_02(Consume, 1) : !aie.objectfifosubview<memref<7x7x256xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<7x7x256xui8>> -> memref<7x7x256xui8>
        %2 = aie.objectfifo.acquire @out_02_L2(Produce, 1) : !aie.objectfifosubview<memref<1x1x256xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1x1x256xui8>> -> memref<1x1x256xui8>
        %c7_i32 = arith.constant 7 : i32
        %c7_i32_0 = arith.constant 7 : i32
        %c256_i32 = arith.constant 256 : i32
        %c7_i32_1 = arith.constant 7 : i32
        func.call @average_pooling(%1, %3, %c7_i32, %c7_i32_0, %c256_i32, %c7_i32_1) : (memref<7x7x256xui8>, memref<1x1x256xui8>, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_L2_02(Consume, 1)
        aie.objectfifo.release @out_02_L2(Produce, 1)
      }
      aie.end
    } {link_with = "avg_pool.o"}
    func.func @sequence(%arg0: memref<3136xi32>, %arg1: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 3136][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<3136xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0]) {id = 1 : i64, metadata = @outOFL2L3} : memref<64xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

