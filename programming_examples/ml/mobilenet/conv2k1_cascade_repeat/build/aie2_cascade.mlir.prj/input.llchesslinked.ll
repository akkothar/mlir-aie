; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target triple = "aie2"

%struct.ipd.custom_type.uint2_t.uint2_t = type { i2 }

@rtp04 = external global [16 x i32]
@inOF_act_L3L2_1_cons_buff_1 = external global [7 x [1 x [64 x i8]]]
@inOF_act_L3L2_1_cons_buff_0 = external global [7 x [1 x [64 x i8]]]
@inOF_act_L3L2_0_cons_buff_1 = external global [7 x [1 x [64 x i8]]]
@inOF_act_L3L2_0_cons_buff_0 = external global [7 x [1 x [64 x i8]]]
@OF_wts_memtile_put_cons_buff_0 = external global [2048 x i8]
@OF_wts_memtile_get_cons_buff_0 = external global [2048 x i8]
@OF_bn_12_buff_1 = external global [7 x [1 x [64 x i8]]]
@OF_bn_12_buff_0 = external global [7 x [1 x [64 x i8]]]

define void @sequence(ptr %0, ptr %1, ptr %2) {
  ret void
}

define void @core_0_4() {
  br label %1

1:                                                ; preds = %42, %0
  %2 = phi i64 [ %47, %42 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967294
  br i1 %3, label %4, label %48

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  %5 = and i64 ptrtoint (ptr @rtp04 to i64), 31
  %6 = icmp eq i64 %5, 0
  call void @llvm.assume(i1 %6)
  %7 = load i32, ptr @rtp04, align 4
  call void @llvm.assume(i1 %6)
  %8 = load i32, ptr getelementptr (i32, ptr @rtp04, i32 1), align 4
  br label %9

9:                                                ; preds = %12, %4
  %10 = phi i64 [ %23, %12 ], [ 0, %4 ]
  %11 = icmp slt i64 %10, 6
  br i1 %11, label %12, label %24

12:                                               ; preds = %9
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %13 = and i64 ptrtoint (ptr @OF_bn_12_buff_0 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  %15 = and i64 ptrtoint (ptr @OF_wts_memtile_get_cons_buff_0 to i64), 31
  %16 = icmp eq i64 %15, 0
  call void @llvm.assume(i1 %16)
  %17 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %18)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_0, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_0, ptr @inOF_act_L3L2_1_cons_buff_0, i32 7, i32 64, i32 64, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %19 = and i64 ptrtoint (ptr @OF_bn_12_buff_1 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @llvm.assume(i1 %16)
  %21 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_1 to i64), 31
  %22 = icmp eq i64 %21, 0
  call void @llvm.assume(i1 %22)
  call void @llvm.assume(i1 %22)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_1, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_1, ptr @inOF_act_L3L2_1_cons_buff_1, i32 7, i32 64, i32 64, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %23 = add i64 %10, 2
  br label %9

24:                                               ; preds = %9
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %25 = and i64 ptrtoint (ptr @OF_bn_12_buff_0 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  %27 = and i64 ptrtoint (ptr @OF_wts_memtile_get_cons_buff_0 to i64), 31
  %28 = icmp eq i64 %27, 0
  call void @llvm.assume(i1 %28)
  %29 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_0 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_0, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_0, ptr @inOF_act_L3L2_1_cons_buff_0, i32 7, i32 64, i32 64, i32 %7, i32 %8)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  call void @llvm.assume(i1 %6)
  %31 = load i32, ptr @rtp04, align 4
  call void @llvm.assume(i1 %6)
  %32 = load i32, ptr getelementptr (i32, ptr @rtp04, i32 1), align 4
  br label %33

33:                                               ; preds = %36, %24
  %34 = phi i64 [ %41, %36 ], [ 0, %24 ]
  %35 = icmp slt i64 %34, 6
  br i1 %35, label %36, label %42

36:                                               ; preds = %33
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %37 = and i64 ptrtoint (ptr @OF_bn_12_buff_1 to i64), 31
  %38 = icmp eq i64 %37, 0
  call void @llvm.assume(i1 %38)
  call void @llvm.assume(i1 %28)
  %39 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_1 to i64), 31
  %40 = icmp eq i64 %39, 0
  call void @llvm.assume(i1 %40)
  call void @llvm.assume(i1 %40)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_1, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_1, ptr @inOF_act_L3L2_1_cons_buff_1, i32 7, i32 64, i32 64, i32 %31, i32 %32)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  call void @llvm.assume(i1 %26)
  call void @llvm.assume(i1 %28)
  call void @llvm.assume(i1 %30)
  call void @llvm.assume(i1 %30)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_0, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_0, ptr @inOF_act_L3L2_1_cons_buff_0, i32 7, i32 64, i32 64, i32 %31, i32 %32)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %41 = add i64 %34, 2
  br label %33

42:                                               ; preds = %33
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %43 = and i64 ptrtoint (ptr @OF_bn_12_buff_1 to i64), 31
  %44 = icmp eq i64 %43, 0
  call void @llvm.assume(i1 %44)
  call void @llvm.assume(i1 %28)
  %45 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_1 to i64), 31
  %46 = icmp eq i64 %45, 0
  call void @llvm.assume(i1 %46)
  call void @llvm.assume(i1 %46)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_1, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_1, ptr @inOF_act_L3L2_1_cons_buff_1, i32 7, i32 64, i32 64, i32 %31, i32 %32)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %47 = add i64 %2, 2
  br label %1

48:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  %49 = and i64 ptrtoint (ptr @rtp04 to i64), 31
  %50 = icmp eq i64 %49, 0
  call void @llvm.assume(i1 %50)
  %51 = load i32, ptr @rtp04, align 4
  call void @llvm.assume(i1 %50)
  %52 = load i32, ptr getelementptr (i32, ptr @rtp04, i32 1), align 4
  br label %53

53:                                               ; preds = %56, %48
  %54 = phi i64 [ %67, %56 ], [ 0, %48 ]
  %55 = icmp slt i64 %54, 6
  br i1 %55, label %56, label %68

56:                                               ; preds = %53
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %57 = and i64 ptrtoint (ptr @OF_bn_12_buff_0 to i64), 31
  %58 = icmp eq i64 %57, 0
  call void @llvm.assume(i1 %58)
  %59 = and i64 ptrtoint (ptr @OF_wts_memtile_get_cons_buff_0 to i64), 31
  %60 = icmp eq i64 %59, 0
  call void @llvm.assume(i1 %60)
  %61 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_0 to i64), 31
  %62 = icmp eq i64 %61, 0
  call void @llvm.assume(i1 %62)
  call void @llvm.assume(i1 %62)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_0, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_0, ptr @inOF_act_L3L2_1_cons_buff_0, i32 7, i32 64, i32 64, i32 %51, i32 %52)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %63 = and i64 ptrtoint (ptr @OF_bn_12_buff_1 to i64), 31
  %64 = icmp eq i64 %63, 0
  call void @llvm.assume(i1 %64)
  call void @llvm.assume(i1 %60)
  %65 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_1 to i64), 31
  %66 = icmp eq i64 %65, 0
  call void @llvm.assume(i1 %66)
  call void @llvm.assume(i1 %66)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_1, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_1, ptr @inOF_act_L3L2_1_cons_buff_1, i32 7, i32 64, i32 64, i32 %51, i32 %52)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  %67 = add i64 %54, 2
  br label %53

68:                                               ; preds = %53
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.aie2.acquire(i32 52, i32 -1)
  %69 = and i64 ptrtoint (ptr @OF_bn_12_buff_0 to i64), 31
  %70 = icmp eq i64 %69, 0
  call void @llvm.assume(i1 %70)
  %71 = and i64 ptrtoint (ptr @OF_wts_memtile_get_cons_buff_0 to i64), 31
  %72 = icmp eq i64 %71, 0
  call void @llvm.assume(i1 %72)
  %73 = and i64 ptrtoint (ptr @inOF_act_L3L2_1_cons_buff_0 to i64), 31
  %74 = icmp eq i64 %73, 0
  call void @llvm.assume(i1 %74)
  call void @llvm.assume(i1 %74)
  call void @conv2dk1_skip_ui8_i8_i8_get(ptr @inOF_act_L3L2_1_cons_buff_0, ptr @OF_wts_memtile_get_cons_buff_0, ptr @OF_bn_12_buff_0, ptr @inOF_act_L3L2_1_cons_buff_0, i32 7, i32 64, i32 64, i32 %51, i32 %52)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 53, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  ret void
}

declare void @llvm.aie2.acquire(i32, i32)

; Function Attrs: nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly
declare void @llvm.assume(i1 noundef) #0

declare void @conv2dk1_skip_ui8_i8_i8_get(ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32)

declare void @llvm.aie2.release(i32, i32)

define void @core_0_5() {
  br label %1

1:                                                ; preds = %28, %0
  %2 = phi i64 [ %31, %28 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 4294967294
  br i1 %3, label %4, label %32

4:                                                ; preds = %1
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  br label %5

5:                                                ; preds = %8, %4
  %6 = phi i64 [ %15, %8 ], [ 0, %4 ]
  %7 = icmp slt i64 %6, 6
  br i1 %7, label %8, label %16

8:                                                ; preds = %5
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %9 = and i64 ptrtoint (ptr @OF_wts_memtile_put_cons_buff_0 to i64), 31
  %10 = icmp eq i64 %9, 0
  call void @llvm.assume(i1 %10)
  %11 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_0 to i64), 31
  %12 = icmp eq i64 %11, 0
  call void @llvm.assume(i1 %12)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_0, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 %10)
  %13 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_1 to i64), 31
  %14 = icmp eq i64 %13, 0
  call void @llvm.assume(i1 %14)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_1, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  %15 = add i64 %6, 2
  br label %5

16:                                               ; preds = %5
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %17 = and i64 ptrtoint (ptr @OF_wts_memtile_put_cons_buff_0 to i64), 31
  %18 = icmp eq i64 %17, 0
  call void @llvm.assume(i1 %18)
  %19 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_0 to i64), 31
  %20 = icmp eq i64 %19, 0
  call void @llvm.assume(i1 %20)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_0, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  br label %21

21:                                               ; preds = %24, %16
  %22 = phi i64 [ %27, %24 ], [ 0, %16 ]
  %23 = icmp slt i64 %22, 6
  br i1 %23, label %24, label %28

24:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 %18)
  %25 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_1 to i64), 31
  %26 = icmp eq i64 %25, 0
  call void @llvm.assume(i1 %26)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_1, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 %18)
  call void @llvm.assume(i1 %20)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_0, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  %27 = add i64 %22, 2
  br label %21

28:                                               ; preds = %21
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 %18)
  %29 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_1 to i64), 31
  %30 = icmp eq i64 %29, 0
  call void @llvm.assume(i1 %30)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_1, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  %31 = add i64 %2, 2
  br label %1

32:                                               ; preds = %1
  call void @llvm.aie2.acquire(i32 51, i32 -1)
  br label %33

33:                                               ; preds = %36, %32
  %34 = phi i64 [ %43, %36 ], [ 0, %32 ]
  %35 = icmp slt i64 %34, 6
  br i1 %35, label %36, label %44

36:                                               ; preds = %33
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %37 = and i64 ptrtoint (ptr @OF_wts_memtile_put_cons_buff_0 to i64), 31
  %38 = icmp eq i64 %37, 0
  call void @llvm.assume(i1 %38)
  %39 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_0 to i64), 31
  %40 = icmp eq i64 %39, 0
  call void @llvm.assume(i1 %40)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_0, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  call void @llvm.assume(i1 %38)
  %41 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_1 to i64), 31
  %42 = icmp eq i64 %41, 0
  call void @llvm.assume(i1 %42)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_1, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  %43 = add i64 %34, 2
  br label %33

44:                                               ; preds = %33
  call void @llvm.aie2.acquire(i32 49, i32 -1)
  %45 = and i64 ptrtoint (ptr @OF_wts_memtile_put_cons_buff_0 to i64), 31
  %46 = icmp eq i64 %45, 0
  call void @llvm.assume(i1 %46)
  %47 = and i64 ptrtoint (ptr @inOF_act_L3L2_0_cons_buff_0 to i64), 31
  %48 = icmp eq i64 %47, 0
  call void @llvm.assume(i1 %48)
  call void @conv2dk1_skip_ui8_i8_put(ptr @inOF_act_L3L2_0_cons_buff_0, ptr @OF_wts_memtile_put_cons_buff_0, i32 7, i32 64, i32 64)
  call void @llvm.aie2.release(i32 48, i32 1)
  call void @llvm.aie2.release(i32 50, i32 1)
  ret void
}

declare void @conv2dk1_skip_ui8_i8_put(ptr, ptr, i32, i32, i32)

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___acquire(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext %0, i32 zeroext %1) #5
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #2

; Function Attrs: nounwind inaccessiblememonly
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire_guarded___uint___uint(i32 zeroext, i32 zeroext) local_unnamed_addr addrspace(1) #3

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2___release(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #1 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext %0, i32 signext %1) #5
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #5
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release_guarded___uint___sint(i32 zeroext, i32 signext) local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind
define dso_local void @llvm___aie___event0() local_unnamed_addr addrspace(1) #4 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t zeroinitializer) #5
  ret void
}

; Function Attrs: nounwind inaccessiblememonly
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t) local_unnamed_addr addrspace(1) #3

; Function Attrs: nounwind
define dso_local void @llvm___aie___event1() local_unnamed_addr addrspace(1) #4 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t { i2 1 }) #5
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn inaccessiblememonly writeonly }
attributes #1 = { mustprogress nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { nounwind inaccessiblememonly "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { nounwind inaccessiblememonly "no-builtin-memcpy" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.linker.options = !{}
!llvm.ident = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang version 15.0.5 (/u/sgasip/ipd/repositories/llvm_ipd 3a25925e0239306412dac02da5e4c8c51ae722e8)"}
