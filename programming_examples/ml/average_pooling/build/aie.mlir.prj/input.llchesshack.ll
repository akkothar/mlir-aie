; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie2"

@inOF_act_L3L2_cons_buff_0 = external global [7 x [7 x [256 x i8]]]
@act_L2_02_cons_buff_0 = external global [7 x [7 x [256 x i8]]]
@out_02_L2_buff_0 = external global [1 x [1 x [256 x i8]]]
@out_02_L2_cons_buff_0 = external global [1 x [1 x [256 x i8]]]
@outOFL2L3_cons = external global [1 x [1 x [256 x i8]]]
@outOFL2L3 = external global [1 x [1 x [256 x i8]]]
@out_02_L2_cons = external global [1 x [1 x [256 x i8]]]
@out_02_L2 = external global [1 x [1 x [256 x i8]]]
@act_L2_02_cons = external global [7 x [7 x [256 x i8]]]
@act_L2_02 = external global [7 x [7 x [256 x i8]]]
@inOF_act_L3L2_cons = external global [7 x [7 x [256 x i8]]]
@inOF_act_L3L2 = external global [7 x [7 x [256 x i8]]]

declare void @debug_i32(i32)

declare void @llvm.aie2.put.ms(i32, i32)

declare { i32, i32 } @llvm.aie2.get.ss()

declare void @llvm.aie2.mcd.write.vec(<16 x i32>, i32)

declare <16 x i32> @llvm.aie2.scd.read.vec(i32)

declare void @llvm.aie2.acquire(i32, i32)

declare void @llvm.aie2.release(i32, i32)

declare void @average_pooling(ptr, ptr, i32, i32, i32, i32)

define void @sequence(ptr %0, ptr %1) {
  ret void
}

define void @core_0_2() {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi i64 [ %9, %4 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 9223372036854775807
  br i1 %3, label %4, label %10

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 50, i32 -1)
  %5 = and i64 ptrtoint (ptr @out_02_L2_buff_0 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = and i64 ptrtoint (ptr @act_L2_02_cons_buff_0 to i64), 31
  %8 = icmp eq i64 %7, 0
  call void @llvm.assume(i1 %8)
  call void @average_pooling(ptr @act_L2_02_cons_buff_0, ptr @out_02_L2_buff_0, i32 7, i32 7, i32 256, i32 7)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 51, i32 1)
  %9 = add i64 %2, 1
  br label %1

10:                                               ; preds = %1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
