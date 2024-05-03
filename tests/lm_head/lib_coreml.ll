; ModuleID = 'lib_coreml.mm'
source_filename = "lib_coreml.mm"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx14.0.0"

%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct.__NSConstantString_tag = type { ptr, i32, ptr, i64 }
%"class.std::__1::basic_ostream" = type { ptr, %"class.std::__1::basic_ios.base" }
%"class.std::__1::basic_ios.base" = type <{ %"class.std::__1::ios_base", ptr, i32 }>
%"class.std::__1::ios_base" = type { ptr, i32, i64, i64, i32, i32, ptr, ptr, ptr, ptr, i64, i64, ptr, i64, i64, ptr, i64, i64 }
%struct.__builtin_NSConstantIntegerNumber = type { ptr, ptr, i64 }
%"class.std::__1::locale::id" = type <{ %"struct.std::__1::once_flag", i32, [4 x i8] }>
%"struct.std::__1::once_flag" = type { i64 }
%"class.std::__1::basic_ostream<char>::sentry" = type { i8, ptr }
%"class.std::__1::ostreambuf_iterator" = type { ptr }
%"class.std::__1::basic_string" = type { %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"struct.std::__1::__compressed_pair_elem" }
%"struct.std::__1::__compressed_pair_elem" = type { %"struct.std::__1::basic_string<char>::__rep" }
%"struct.std::__1::basic_string<char>::__rep" = type { %union.anon }
%union.anon = type { %"struct.std::__1::basic_string<char>::__long" }
%"struct.std::__1::basic_string<char>::__long" = type { %struct.anon, i64, ptr }
%struct.anon = type { i64 }
%"class.std::__1::basic_ios" = type <{ %"class.std::__1::ios_base", ptr, i32, [4 x i8] }>
%"struct.std::__1::__default_init_tag" = type { i8 }
%"struct.std::__1::basic_string<char>::__short" = type { %struct.anon.0, [0 x i8], [23 x i8] }
%struct.anon.0 = type { i8 }
%"class.std::__1::locale" = type { ptr }

@lm_head_model = global ptr null, align 8
@qkv_out_proj_model = global ptr null, align 8
@down_proj_model = global ptr null, align 8
@gate_proj_model = global ptr null, align 8
@error = global ptr null, align 8
@"OBJC_CLASS_$_NSString" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_" = internal global ptr @"OBJC_CLASS_$_NSString", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@__CFConstantStringClassReference = external global [0 x i32]
@.str = private unnamed_addr constant [3 x i8] c"%@\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str, i64 2 }, section "__DATA,__cfstring", align 8 #0
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [9 x i8] c"userInfo\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [18 x i8] c"stringWithFormat:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.2 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.1, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.3 = private unnamed_addr constant [11 x i8] c"UTF8String\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.4 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.3, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@_ZNSt3__14coutE = external global %"class.std::__1::basic_ostream", align 8
@_ZTISt13runtime_error = external constant ptr
@"OBJC_CLASS_$_MLMultiArray" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.5" = internal global ptr @"OBJC_CLASS_$_MLMultiArray", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@"OBJC_CLASS_$_NSNumber" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.6" = internal global ptr @"OBJC_CLASS_$_NSNumber", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.7 = private unnamed_addr constant [15 x i8] c"numberWithInt:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.8 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.7, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_NSArray" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.9" = internal global ptr @"OBJC_CLASS_$_NSArray", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.10 = private unnamed_addr constant [24 x i8] c"arrayWithObjects:count:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.11 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.10, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_NSConstantIntegerNumber" = external global %struct._class_t
@.str.12 = private unnamed_addr constant [2 x i8] c"i\00", align 1
@_unnamed_nsconstantintegernumber_ = private constant %struct.__builtin_NSConstantIntegerNumber { ptr @"OBJC_CLASS_$_NSConstantIntegerNumber", ptr @.str.12, i64 1 }, section "__DATA,__objc_intobj,regular,no_dead_strip", align 8 #0
@OBJC_METH_VAR_NAME_.13 = private unnamed_addr constant [62 x i8] c"initWithDataPointer:shape:dataType:strides:deallocator:error:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.14 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.13, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_MLModelConfiguration" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.15" = internal global ptr @"OBJC_CLASS_$_MLModelConfiguration", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.16 = private unnamed_addr constant [17 x i8] c"setComputeUnits:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.17 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.16, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_NSURL" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.18" = internal global ptr @"OBJC_CLASS_$_NSURL", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@.str.19 = private unnamed_addr constant [97 x i8] c"/Users/ishankagrawal/Desktop/v2-TinyChatEngine/tests/lm_head/../coreml/modules/lm_head.mlpackage\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_.20 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.19, i64 96 }, section "__DATA,__cfstring", align 8 #0
@OBJC_METH_VAR_NAME_.21 = private unnamed_addr constant [15 x i8] c"URLWithString:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.22 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.21, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_MLModel" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.23" = internal global ptr @"OBJC_CLASS_$_MLModel", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.24 = private unnamed_addr constant [25 x i8] c"compileModelAtURL:error:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.25 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.24, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.26 = private unnamed_addr constant [44 x i8] c"modelWithContentsOfURL:configuration:error:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.27 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.26, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_MLDictionaryFeatureProvider" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.28" = internal global ptr @"OBJC_CLASS_$_MLDictionaryFeatureProvider", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@.str.29 = private unnamed_addr constant [2 x i8] c"x\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_.30 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.29, i64 1 }, section "__DATA,__cfstring", align 8 #0
@.str.31 = private unnamed_addr constant [2 x i8] c"y\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_.32 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.31, i64 1 }, section "__DATA,__cfstring", align 8 #0
@"OBJC_CLASS_$_NSDictionary" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.33" = internal global ptr @"OBJC_CLASS_$_NSDictionary", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.34 = private unnamed_addr constant [37 x i8] c"dictionaryWithObjects:forKeys:count:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.35 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.34, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@OBJC_METH_VAR_NAME_.36 = private unnamed_addr constant [26 x i8] c"initWithDictionary:error:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.37 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.36, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@"OBJC_CLASS_$_MLPredictionOptions" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_.38" = internal global ptr @"OBJC_CLASS_$_MLPredictionOptions", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@.str.39 = private unnamed_addr constant [7 x i8] c"matmul\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_.40 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.39, i64 6 }, section "__DATA,__cfstring", align 8 #0
@OBJC_METH_VAR_NAME_.41 = private unnamed_addr constant [19 x i8] c"setOutputBackings:\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_SELECTOR_REFERENCES_.42 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.41, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@_ZNSt3__15ctypeIcE2idE = external global %"class.std::__1::locale::id", align 8
@llvm.compiler.used = appending global [36 x ptr] [ptr @"OBJC_CLASSLIST_REFERENCES_$_", ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_SELECTOR_REFERENCES_, ptr @OBJC_METH_VAR_NAME_.1, ptr @OBJC_SELECTOR_REFERENCES_.2, ptr @OBJC_METH_VAR_NAME_.3, ptr @OBJC_SELECTOR_REFERENCES_.4, ptr @"OBJC_CLASSLIST_REFERENCES_$_.5", ptr @"OBJC_CLASSLIST_REFERENCES_$_.6", ptr @OBJC_METH_VAR_NAME_.7, ptr @OBJC_SELECTOR_REFERENCES_.8, ptr @"OBJC_CLASSLIST_REFERENCES_$_.9", ptr @OBJC_METH_VAR_NAME_.10, ptr @OBJC_SELECTOR_REFERENCES_.11, ptr @OBJC_METH_VAR_NAME_.13, ptr @OBJC_SELECTOR_REFERENCES_.14, ptr @"OBJC_CLASSLIST_REFERENCES_$_.15", ptr @OBJC_METH_VAR_NAME_.16, ptr @OBJC_SELECTOR_REFERENCES_.17, ptr @"OBJC_CLASSLIST_REFERENCES_$_.18", ptr @OBJC_METH_VAR_NAME_.21, ptr @OBJC_SELECTOR_REFERENCES_.22, ptr @"OBJC_CLASSLIST_REFERENCES_$_.23", ptr @OBJC_METH_VAR_NAME_.24, ptr @OBJC_SELECTOR_REFERENCES_.25, ptr @OBJC_METH_VAR_NAME_.26, ptr @OBJC_SELECTOR_REFERENCES_.27, ptr @"OBJC_CLASSLIST_REFERENCES_$_.28", ptr @"OBJC_CLASSLIST_REFERENCES_$_.33", ptr @OBJC_METH_VAR_NAME_.34, ptr @OBJC_SELECTOR_REFERENCES_.35, ptr @OBJC_METH_VAR_NAME_.36, ptr @OBJC_SELECTOR_REFERENCES_.37, ptr @"OBJC_CLASSLIST_REFERENCES_$_.38", ptr @OBJC_METH_VAR_NAME_.41, ptr @OBJC_SELECTOR_REFERENCES_.42], section "llvm.metadata"

; Function Attrs: noinline optnone ssp uwtable
define void @_Z13handle_errorsP7NSError(ptr noundef %0) #1 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = icmp ne ptr %6, null
  br i1 %7, label %8, label %27

8:                                                ; preds = %1
  %9 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  %10 = load ptr, ptr %2, align 8
  %11 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load !12
  %12 = call ptr @objc_msgSend(ptr noundef %10, ptr noundef %11)
  %13 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.2, align 8, !invariant.load !12
  %14 = call ptr (ptr, ptr, ptr, ...) @objc_msgSend(ptr noundef %9, ptr noundef %13, ptr noundef @_unnamed_cfstring_, ptr noundef %12)
  %15 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.4, align 8, !invariant.load !12
  %16 = call ptr @objc_msgSend(ptr noundef %14, ptr noundef %15)
  store ptr %16, ptr %3, align 8
  %17 = load ptr, ptr %3, align 8
  %18 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__1lsB7v160006INS_11char_traitsIcEEEERNS_13basic_ostreamIcT_EES6_PKc(ptr noundef nonnull align 8 dereferenceable(8) @_ZNSt3__14coutE, ptr noundef %17)
  %19 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsB7v160006EPFRS3_S4_E(ptr noundef %18, ptr noundef @_ZNSt3__14endlB7v160006IcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_)
  %20 = call ptr @__cxa_allocate_exception(i64 16) #8
  %21 = load ptr, ptr %3, align 8
  invoke void @_ZNSt13runtime_errorC1EPKc(ptr noundef %20, ptr noundef %21)
          to label %22 unwind label %23

22:                                               ; preds = %8
  call void @__cxa_throw(ptr %20, ptr @_ZTISt13runtime_error, ptr @_ZNSt13runtime_errorD1Ev) #9
  unreachable

23:                                               ; preds = %8
  %24 = landingpad { ptr, i32 }
          cleanup
  %25 = extractvalue { ptr, i32 } %24, 0
  store ptr %25, ptr %4, align 8
  %26 = extractvalue { ptr, i32 } %24, 1
  store i32 %26, ptr %5, align 4
  call void @__cxa_free_exception(ptr %20) #8
  br label %28

27:                                               ; preds = %1
  ret void

28:                                               ; preds = %23
  %29 = load ptr, ptr %4, align 8
  %30 = load i32, ptr %5, align 4
  %31 = insertvalue { ptr, i32 } undef, ptr %29, 0
  %32 = insertvalue { ptr, i32 } %31, i32 %30, 1
  resume { ptr, i32 } %32
}

; Function Attrs: nonlazybind
declare ptr @objc_msgSend(ptr, ptr, ...) #2

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__1lsB7v160006INS_11char_traitsIcEEEERNS_13basic_ostreamIcT_EES6_PKc(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1) #1 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = call i64 @_ZNSt3__111char_traitsIcE6lengthEPKc(ptr noundef %7) #8
  %9 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__124__put_character_sequenceB7v160006IcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m(ptr noundef nonnull align 8 dereferenceable(8) %5, ptr noundef %6, i64 noundef %8)
  ret ptr %9
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEElsB7v160006EPFRS3_S4_E(ptr noundef %0, ptr noundef %1) #1 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = call noundef nonnull align 8 dereferenceable(8) ptr %6(ptr noundef nonnull align 8 dereferenceable(8) %5)
  ret ptr %7
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__14endlB7v160006IcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_(ptr noundef nonnull align 8 dereferenceable(8) %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr i8, ptr %5, i64 -24
  %7 = load i64, ptr %6, align 8
  %8 = getelementptr inbounds i8, ptr %4, i64 %7
  %9 = call signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenB7v160006Ec(ptr noundef %8, i8 noundef signext 10)
  %10 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE3putEc(ptr noundef %3, i8 noundef signext %9)
  %11 = load ptr, ptr %2, align 8
  %12 = call noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE5flushEv(ptr noundef %11)
  %13 = load ptr, ptr %2, align 8
  ret ptr %13
}

declare ptr @__cxa_allocate_exception(i64)

declare void @_ZNSt13runtime_errorC1EPKc(ptr noundef, ptr noundef) unnamed_addr #3

declare void @__cxa_free_exception(ptr)

; Function Attrs: nounwind
declare void @_ZNSt13runtime_errorD1Ev(ptr noundef) unnamed_addr #4

declare void @__cxa_throw(ptr, ptr, ptr)

; Function Attrs: noinline optnone ssp uwtable
define ptr @_Z21float_to_MLMultiArrayPKfiiP7NSError(ptr noalias noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %3) #1 {
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca [2 x ptr], align 8
  %11 = alloca [2 x ptr], align 8
  store ptr %0, ptr %5, align 8
  store i32 %1, ptr %6, align 4
  store i32 %2, ptr %7, align 4
  store ptr %3, ptr %8, align 8
  %12 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.5", align 8
  %13 = call ptr @objc_alloc(ptr %12)
  %14 = load ptr, ptr %5, align 8
  %15 = getelementptr inbounds [2 x ptr], ptr %10, i64 0, i64 0
  %16 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.6", align 8
  %17 = load i32, ptr %6, align 4
  %18 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.8, align 8, !invariant.load !12
  %19 = call ptr @objc_msgSend(ptr noundef %16, ptr noundef %18, i32 noundef %17)
  store ptr %19, ptr %15, align 8
  %20 = getelementptr inbounds [2 x ptr], ptr %10, i64 0, i64 1
  %21 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.6", align 8
  %22 = load i32, ptr %7, align 4
  %23 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.8, align 8, !invariant.load !12
  %24 = call ptr @objc_msgSend(ptr noundef %21, ptr noundef %23, i32 noundef %22)
  store ptr %24, ptr %20, align 8
  %25 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.9", align 8
  %26 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.11, align 8, !invariant.load !12
  %27 = call ptr @objc_msgSend(ptr noundef %25, ptr noundef %26, ptr noundef %10, i64 noundef 2)
  %28 = getelementptr inbounds [2 x ptr], ptr %11, i64 0, i64 0
  %29 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.6", align 8
  %30 = load i32, ptr %7, align 4
  %31 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.8, align 8, !invariant.load !12
  %32 = call ptr @objc_msgSend(ptr noundef %29, ptr noundef %31, i32 noundef %30)
  store ptr %32, ptr %28, align 8
  %33 = getelementptr inbounds [2 x ptr], ptr %11, i64 0, i64 1
  store ptr @_unnamed_nsconstantintegernumber_, ptr %33, align 8
  %34 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.9", align 8
  %35 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.11, align 8, !invariant.load !12
  %36 = call ptr @objc_msgSend(ptr noundef %34, ptr noundef %35, ptr noundef %11, i64 noundef 2)
  %37 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.14, align 8, !invariant.load !12
  %38 = call ptr @objc_msgSend(ptr noundef %13, ptr noundef %37, ptr noundef %14, ptr noundef %27, i64 noundef 65568, ptr noundef %36, ptr noundef null, ptr noundef %8)
  store ptr %38, ptr %9, align 8
  %39 = load ptr, ptr %9, align 8
  ret ptr %39
}

declare ptr @objc_alloc(ptr)

; Function Attrs: noinline optnone ssp uwtable
define void @_Z4initv() #1 {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.15", align 8
  %5 = call ptr @objc_opt_new(ptr %4)
  store ptr %5, ptr %3, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.17, align 8, !invariant.load !12
  call void @objc_msgSend(ptr noundef %6, ptr noundef %7, i64 noundef 3)
  %8 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.18", align 8
  %9 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.22, align 8, !invariant.load !12
  %10 = call ptr @objc_msgSend(ptr noundef %8, ptr noundef %9, ptr noundef @_unnamed_cfstring_.20)
  store ptr %10, ptr %1, align 8
  %11 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.23", align 8
  %12 = load ptr, ptr %1, align 8
  %13 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.25, align 8, !invariant.load !12
  %14 = call ptr @objc_msgSend(ptr noundef %11, ptr noundef %13, ptr noundef %12, ptr noundef @error)
  store ptr %14, ptr %2, align 8
  %15 = load ptr, ptr @error, align 8
  call void @_Z13handle_errorsP7NSError(ptr noundef %15)
  %16 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.23", align 8
  %17 = load ptr, ptr %2, align 8
  %18 = load ptr, ptr %3, align 8
  %19 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.27, align 8, !invariant.load !12
  %20 = call ptr @objc_msgSend(ptr noundef %16, ptr noundef %19, ptr noundef %17, ptr noundef %18, ptr noundef @error)
  store ptr %20, ptr @lm_head_model, align 8
  ret void
}

declare ptr @objc_opt_new(ptr)

; Function Attrs: noinline nounwind optnone ssp uwtable
define void @_Z10do_nothingP27MLDictionaryFeatureProviderP19MLPredictionOptions(ptr noundef %0, ptr noundef %1) #5 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  ret void
}

; Function Attrs: noinline optnone ssp uwtable
define void @_Z13static_matmulPKfS0_S0_iii(ptr noalias noundef %0, ptr noalias noundef %1, ptr noalias noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5) #1 {
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = alloca ptr, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca ptr, align 8
  %14 = alloca ptr, align 8
  %15 = alloca ptr, align 8
  %16 = alloca ptr, align 8
  %17 = alloca [2 x ptr], align 8
  %18 = alloca [2 x ptr], align 8
  %19 = alloca ptr, align 8
  %20 = alloca [1 x ptr], align 8
  %21 = alloca [1 x ptr], align 8
  store ptr %0, ptr %7, align 8
  store ptr %1, ptr %8, align 8
  store ptr %2, ptr %9, align 8
  store i32 %3, ptr %10, align 4
  store i32 %4, ptr %11, align 4
  store i32 %5, ptr %12, align 4
  %22 = load ptr, ptr %7, align 8
  %23 = load i32, ptr %10, align 4
  %24 = load i32, ptr %12, align 4
  %25 = load ptr, ptr @error, align 8
  %26 = call ptr @_Z21float_to_MLMultiArrayPKfiiP7NSError(ptr noundef %22, i32 noundef %23, i32 noundef %24, ptr noundef %25)
  store ptr %26, ptr %13, align 8
  %27 = load ptr, ptr %8, align 8
  %28 = load i32, ptr %11, align 4
  %29 = load i32, ptr %12, align 4
  %30 = load ptr, ptr @error, align 8
  %31 = call ptr @_Z21float_to_MLMultiArrayPKfiiP7NSError(ptr noundef %27, i32 noundef %28, i32 noundef %29, ptr noundef %30)
  store ptr %31, ptr %14, align 8
  %32 = load ptr, ptr %9, align 8
  %33 = load i32, ptr %10, align 4
  %34 = load i32, ptr %11, align 4
  %35 = load ptr, ptr @error, align 8
  %36 = call ptr @_Z21float_to_MLMultiArrayPKfiiP7NSError(ptr noundef %32, i32 noundef %33, i32 noundef %34, ptr noundef %35)
  store ptr %36, ptr %15, align 8
  %37 = load ptr, ptr @error, align 8
  call void @_Z13handle_errorsP7NSError(ptr noundef %37)
  %38 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.28", align 8
  %39 = call ptr @objc_alloc(ptr %38)
  %40 = getelementptr inbounds [2 x ptr], ptr %18, i64 0, i64 0
  store ptr @_unnamed_cfstring_.30, ptr %40, align 8
  %41 = getelementptr inbounds [2 x ptr], ptr %17, i64 0, i64 0
  %42 = load ptr, ptr %13, align 8
  store ptr %42, ptr %41, align 8
  %43 = getelementptr inbounds [2 x ptr], ptr %18, i64 0, i64 1
  store ptr @_unnamed_cfstring_.32, ptr %43, align 8
  %44 = getelementptr inbounds [2 x ptr], ptr %17, i64 0, i64 1
  %45 = load ptr, ptr %14, align 8
  store ptr %45, ptr %44, align 8
  %46 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.33", align 8
  %47 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.35, align 8, !invariant.load !12
  %48 = call ptr @objc_msgSend(ptr noundef %46, ptr noundef %47, ptr noundef %17, ptr noundef %18, i64 noundef 2)
  %49 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.37, align 8, !invariant.load !12
  %50 = call ptr @objc_msgSend(ptr noundef %39, ptr noundef %49, ptr noundef %48, ptr noundef @error)
  store ptr %50, ptr %16, align 8
  %51 = load ptr, ptr @error, align 8
  call void @_Z13handle_errorsP7NSError(ptr noundef %51)
  %52 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.38", align 8
  %53 = call ptr @objc_alloc(ptr %52)
  store ptr %53, ptr %19, align 8
  %54 = getelementptr inbounds [1 x ptr], ptr %21, i64 0, i64 0
  store ptr @_unnamed_cfstring_.40, ptr %54, align 8
  %55 = getelementptr inbounds [1 x ptr], ptr %20, i64 0, i64 0
  %56 = load ptr, ptr %15, align 8
  store ptr %56, ptr %55, align 8
  %57 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.33", align 8
  %58 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.35, align 8, !invariant.load !12
  %59 = call ptr @objc_msgSend(ptr noundef %57, ptr noundef %58, ptr noundef %20, ptr noundef %21, i64 noundef 1)
  %60 = load ptr, ptr %19, align 8
  %61 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.42, align 8, !invariant.load !12
  call void @objc_msgSend(ptr noundef %60, ptr noundef %61, ptr noundef %59)
  %62 = load ptr, ptr %16, align 8
  %63 = load ptr, ptr %19, align 8
  call void @_Z10do_nothingP27MLDictionaryFeatureProviderP19MLPredictionOptions(ptr noundef %62, ptr noundef %63)
  ret void
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__124__put_character_sequenceB7v160006IcNS_11char_traitsIcEEEERNS_13basic_ostreamIT_T0_EES7_PKS4_m(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef %1, i64 noundef %2) #1 personality ptr @__gxx_personality_v0 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca %"class.std::__1::basic_ostream<char>::sentry", align 8
  %8 = alloca ptr, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  %11 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  %12 = load ptr, ptr %4, align 8
  invoke void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_(ptr noundef %7, ptr noundef nonnull align 8 dereferenceable(8) %12)
          to label %13 unwind label %64

13:                                               ; preds = %3
  %14 = invoke zeroext i1 @_ZNKSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentrycvbB7v160006Ev(ptr noundef %7)
          to label %15 unwind label %68

15:                                               ; preds = %13
  br i1 %14, label %16, label %73

16:                                               ; preds = %15
  %17 = load ptr, ptr %4, align 8
  call void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC1B7v160006ERNS_13basic_ostreamIcS2_EE(ptr noundef %11, ptr noundef nonnull align 8 dereferenceable(8) %17) #8
  %18 = load ptr, ptr %5, align 8
  %19 = load ptr, ptr %4, align 8
  %20 = load ptr, ptr %19, align 8
  %21 = getelementptr i8, ptr %20, i64 -24
  %22 = load i64, ptr %21, align 8
  %23 = getelementptr inbounds i8, ptr %19, i64 %22
  %24 = invoke i32 @_ZNKSt3__18ios_base5flagsB7v160006Ev(ptr noundef %23)
          to label %25 unwind label %68

25:                                               ; preds = %16
  %26 = and i32 %24, 176
  %27 = icmp eq i32 %26, 32
  br i1 %27, label %28, label %32

28:                                               ; preds = %25
  %29 = load ptr, ptr %5, align 8
  %30 = load i64, ptr %6, align 8
  %31 = getelementptr inbounds i8, ptr %29, i64 %30
  br label %34

32:                                               ; preds = %25
  %33 = load ptr, ptr %5, align 8
  br label %34

34:                                               ; preds = %32, %28
  %35 = phi ptr [ %31, %28 ], [ %33, %32 ]
  %36 = load ptr, ptr %5, align 8
  %37 = load i64, ptr %6, align 8
  %38 = getelementptr inbounds i8, ptr %36, i64 %37
  %39 = load ptr, ptr %4, align 8
  %40 = load ptr, ptr %39, align 8
  %41 = getelementptr i8, ptr %40, i64 -24
  %42 = load i64, ptr %41, align 8
  %43 = getelementptr inbounds i8, ptr %39, i64 %42
  %44 = load ptr, ptr %4, align 8
  %45 = load ptr, ptr %44, align 8
  %46 = getelementptr i8, ptr %45, i64 -24
  %47 = load i64, ptr %46, align 8
  %48 = getelementptr inbounds i8, ptr %44, i64 %47
  %49 = invoke signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE4fillB7v160006Ev(ptr noundef %48)
          to label %50 unwind label %68

50:                                               ; preds = %34
  %51 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %11, i32 0, i32 0
  %52 = load ptr, ptr %51, align 8
  %53 = invoke ptr @_ZNSt3__116__pad_and_outputB7v160006IcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_(ptr %52, ptr noundef %18, ptr noundef %35, ptr noundef %38, ptr noundef nonnull align 8 dereferenceable(136) %43, i8 noundef signext %49)
          to label %54 unwind label %68

54:                                               ; preds = %50
  %55 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %10, i32 0, i32 0
  store ptr %53, ptr %55, align 8
  %56 = call zeroext i1 @_ZNKSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEE6failedB7v160006Ev(ptr noundef %10) #8
  br i1 %56, label %57, label %72

57:                                               ; preds = %54
  %58 = load ptr, ptr %4, align 8
  %59 = load ptr, ptr %58, align 8
  %60 = getelementptr i8, ptr %59, i64 -24
  %61 = load i64, ptr %60, align 8
  %62 = getelementptr inbounds i8, ptr %58, i64 %61
  invoke void @_ZNSt3__19basic_iosIcNS_11char_traitsIcEEE8setstateB7v160006Ej(ptr noundef %62, i32 noundef 5)
          to label %63 unwind label %68

63:                                               ; preds = %57
  br label %72

64:                                               ; preds = %73, %3
  %65 = landingpad { ptr, i32 }
          catch ptr null
  %66 = extractvalue { ptr, i32 } %65, 0
  store ptr %66, ptr %8, align 8
  %67 = extractvalue { ptr, i32 } %65, 1
  store i32 %67, ptr %9, align 4
  br label %76

68:                                               ; preds = %57, %50, %34, %16, %13
  %69 = landingpad { ptr, i32 }
          catch ptr null
  %70 = extractvalue { ptr, i32 } %69, 0
  store ptr %70, ptr %8, align 8
  %71 = extractvalue { ptr, i32 } %69, 1
  store i32 %71, ptr %9, align 4
  invoke void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev(ptr noundef %7)
          to label %75 unwind label %97

72:                                               ; preds = %63, %54
  br label %73

73:                                               ; preds = %72, %15
  invoke void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev(ptr noundef %7)
          to label %74 unwind label %64

74:                                               ; preds = %73
  br label %85

75:                                               ; preds = %68
  br label %76

76:                                               ; preds = %75, %64
  %77 = load ptr, ptr %8, align 8
  %78 = call ptr @__cxa_begin_catch(ptr %77) #8
  %79 = load ptr, ptr %4, align 8
  %80 = load ptr, ptr %79, align 8
  %81 = getelementptr i8, ptr %80, i64 -24
  %82 = load i64, ptr %81, align 8
  %83 = getelementptr inbounds i8, ptr %79, i64 %82
  invoke void @_ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv(ptr noundef %83)
          to label %84 unwind label %87

84:                                               ; preds = %76
  call void @__cxa_end_catch()
  br label %85

85:                                               ; preds = %84, %74
  %86 = load ptr, ptr %4, align 8
  ret ptr %86

87:                                               ; preds = %76
  %88 = landingpad { ptr, i32 }
          cleanup
  %89 = extractvalue { ptr, i32 } %88, 0
  store ptr %89, ptr %8, align 8
  %90 = extractvalue { ptr, i32 } %88, 1
  store i32 %90, ptr %9, align 4
  invoke void @__cxa_end_catch()
          to label %91 unwind label %97

91:                                               ; preds = %87
  br label %92

92:                                               ; preds = %91
  %93 = load ptr, ptr %8, align 8
  %94 = load i32, ptr %9, align 4
  %95 = insertvalue { ptr, i32 } undef, ptr %93, 0
  %96 = insertvalue { ptr, i32 } %95, i32 %94, 1
  resume { ptr, i32 } %96

97:                                               ; preds = %87, %68
  %98 = landingpad { ptr, i32 }
          catch ptr null
  %99 = extractvalue { ptr, i32 } %98, 0
  call void @__clang_call_terminate(ptr %99) #10
  unreachable
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr i64 @_ZNSt3__111char_traitsIcE6lengthEPKc(ptr noundef %0) #5 align 2 personality ptr @__gxx_personality_v0 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %2, align 8
  %5 = load ptr, ptr %2, align 8
  %6 = invoke i64 @_ZNSt3__118__constexpr_strlenB7v160006EPKc(ptr noundef %5)
          to label %7 unwind label %8

7:                                                ; preds = %1
  ret i64 %6

8:                                                ; preds = %1
  %9 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %10 = extractvalue { ptr, i32 } %9, 0
  store ptr %10, ptr %3, align 8
  %11 = extractvalue { ptr, i32 } %9, 1
  store i32 %11, ptr %4, align 4
  br label %12

12:                                               ; preds = %8
  %13 = load ptr, ptr %3, align 8
  call void @__cxa_call_unexpected(ptr %13) #9
  unreachable
}

declare void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryC1ERS3_(ptr noundef, ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #3

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden zeroext i1 @_ZNKSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentrycvbB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::basic_ostream<char>::sentry", ptr %3, i32 0, i32 0
  %5 = load i8, ptr %4, align 8
  %6 = trunc i8 %5 to i1
  ret i1 %6
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNSt3__116__pad_and_outputB7v160006IcNS_11char_traitsIcEEEENS_19ostreambuf_iteratorIT_T0_EES6_PKS4_S8_S8_RNS_8ios_baseES4_(ptr %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef nonnull align 8 dereferenceable(136) %4, i8 noundef signext %5) #1 personality ptr @__gxx_personality_v0 {
  %7 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  %8 = alloca %"class.std::__1::ostreambuf_iterator", align 8
  %9 = alloca ptr, align 8
  %10 = alloca ptr, align 8
  %11 = alloca ptr, align 8
  %12 = alloca ptr, align 8
  %13 = alloca i8, align 1
  %14 = alloca i64, align 8
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca %"class.std::__1::basic_string", align 8
  %18 = alloca ptr, align 8
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  store ptr %0, ptr %21, align 8
  store ptr %1, ptr %9, align 8
  store ptr %2, ptr %10, align 8
  store ptr %3, ptr %11, align 8
  store ptr %4, ptr %12, align 8
  store i8 %5, ptr %13, align 1
  %22 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  %23 = load ptr, ptr %22, align 8
  %24 = icmp eq ptr %23, null
  br i1 %24, label %25, label %26

25:                                               ; preds = %6
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %8, i64 8, i1 false)
  br label %108

26:                                               ; preds = %6
  %27 = load ptr, ptr %11, align 8
  %28 = load ptr, ptr %9, align 8
  %29 = ptrtoint ptr %27 to i64
  %30 = ptrtoint ptr %28 to i64
  %31 = sub i64 %29, %30
  store i64 %31, ptr %14, align 8
  %32 = load ptr, ptr %12, align 8
  %33 = call i64 @_ZNKSt3__18ios_base5widthB7v160006Ev(ptr noundef %32)
  store i64 %33, ptr %15, align 8
  %34 = load i64, ptr %15, align 8
  %35 = load i64, ptr %14, align 8
  %36 = icmp sgt i64 %34, %35
  br i1 %36, label %37, label %41

37:                                               ; preds = %26
  %38 = load i64, ptr %14, align 8
  %39 = load i64, ptr %15, align 8
  %40 = sub nsw i64 %39, %38
  store i64 %40, ptr %15, align 8
  br label %42

41:                                               ; preds = %26
  store i64 0, ptr %15, align 8
  br label %42

42:                                               ; preds = %41, %37
  %43 = load ptr, ptr %10, align 8
  %44 = load ptr, ptr %9, align 8
  %45 = ptrtoint ptr %43 to i64
  %46 = ptrtoint ptr %44 to i64
  %47 = sub i64 %45, %46
  store i64 %47, ptr %16, align 8
  %48 = load i64, ptr %16, align 8
  %49 = icmp sgt i64 %48, 0
  br i1 %49, label %50, label %61

50:                                               ; preds = %42
  %51 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  %52 = load ptr, ptr %51, align 8
  %53 = load ptr, ptr %9, align 8
  %54 = load i64, ptr %16, align 8
  %55 = call i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnB7v160006EPKcl(ptr noundef %52, ptr noundef %53, i64 noundef %54)
  %56 = load i64, ptr %16, align 8
  %57 = icmp ne i64 %55, %56
  br i1 %57, label %58, label %60

58:                                               ; preds = %50
  %59 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  store ptr null, ptr %59, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %8, i64 8, i1 false)
  br label %108

60:                                               ; preds = %50
  br label %61

61:                                               ; preds = %60, %42
  %62 = load i64, ptr %15, align 8
  %63 = icmp sgt i64 %62, 0
  br i1 %63, label %64, label %86

64:                                               ; preds = %61
  %65 = load i64, ptr %15, align 8
  %66 = load i8, ptr %13, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1B7v160006Emc(ptr noundef %17, i64 noundef %65, i8 noundef signext %66)
  %67 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  %68 = load ptr, ptr %67, align 8
  %69 = call ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataB7v160006Ev(ptr noundef %17) #8
  %70 = load i64, ptr %15, align 8
  %71 = invoke i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnB7v160006EPKcl(ptr noundef %68, ptr noundef %69, i64 noundef %70)
          to label %72 unwind label %77

72:                                               ; preds = %64
  %73 = load i64, ptr %15, align 8
  %74 = icmp ne i64 %71, %73
  br i1 %74, label %75, label %81

75:                                               ; preds = %72
  %76 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  store ptr null, ptr %76, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %8, i64 8, i1 false)
  store i32 1, ptr %20, align 4
  br label %82

77:                                               ; preds = %64
  %78 = landingpad { ptr, i32 }
          cleanup
  %79 = extractvalue { ptr, i32 } %78, 0
  store ptr %79, ptr %18, align 8
  %80 = extractvalue { ptr, i32 } %78, 1
  store i32 %80, ptr %19, align 4
  invoke void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr noundef %17)
          to label %85 unwind label %116

81:                                               ; preds = %72
  store i32 0, ptr %20, align 4
  br label %82

82:                                               ; preds = %81, %75
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr noundef %17)
  %83 = load i32, ptr %20, align 4
  switch i32 %83, label %119 [
    i32 0, label %84
    i32 1, label %108
  ]

84:                                               ; preds = %82
  br label %86

85:                                               ; preds = %77
  br label %111

86:                                               ; preds = %84, %61
  %87 = load ptr, ptr %11, align 8
  %88 = load ptr, ptr %10, align 8
  %89 = ptrtoint ptr %87 to i64
  %90 = ptrtoint ptr %88 to i64
  %91 = sub i64 %89, %90
  store i64 %91, ptr %16, align 8
  %92 = load i64, ptr %16, align 8
  %93 = icmp sgt i64 %92, 0
  br i1 %93, label %94, label %105

94:                                               ; preds = %86
  %95 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  %96 = load ptr, ptr %95, align 8
  %97 = load ptr, ptr %10, align 8
  %98 = load i64, ptr %16, align 8
  %99 = call i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnB7v160006EPKcl(ptr noundef %96, ptr noundef %97, i64 noundef %98)
  %100 = load i64, ptr %16, align 8
  %101 = icmp ne i64 %99, %100
  br i1 %101, label %102, label %104

102:                                              ; preds = %94
  %103 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %8, i32 0, i32 0
  store ptr null, ptr %103, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %8, i64 8, i1 false)
  br label %108

104:                                              ; preds = %94
  br label %105

105:                                              ; preds = %104, %86
  %106 = load ptr, ptr %12, align 8
  %107 = call i64 @_ZNSt3__18ios_base5widthB7v160006El(ptr noundef %106, i64 noundef 0)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %7, ptr align 8 %8, i64 8, i1 false)
  br label %108

108:                                              ; preds = %105, %102, %82, %58, %25
  %109 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %7, i32 0, i32 0
  %110 = load ptr, ptr %109, align 8
  ret ptr %110

111:                                              ; preds = %85
  %112 = load ptr, ptr %18, align 8
  %113 = load i32, ptr %19, align 4
  %114 = insertvalue { ptr, i32 } undef, ptr %112, 0
  %115 = insertvalue { ptr, i32 } %114, i32 %113, 1
  resume { ptr, i32 } %115

116:                                              ; preds = %77
  %117 = landingpad { ptr, i32 }
          catch ptr null
  %118 = extractvalue { ptr, i32 } %117, 0
  call void @__clang_call_terminate(ptr %118) #10
  unreachable

119:                                              ; preds = %82
  unreachable
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC1B7v160006ERNS_13basic_ostreamIcS2_EE(ptr noundef %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #5 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = load ptr, ptr %3, align 8
  %6 = load ptr, ptr %4, align 8
  call void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC2B7v160006ERNS_13basic_ostreamIcS2_EE(ptr noundef %5, ptr noundef nonnull align 8 dereferenceable(8) %6) #8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden i32 @_ZNKSt3__18ios_base5flagsB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ios_base", ptr %3, i32 0, i32 1
  %5 = load i32, ptr %4, align 8
  ret i32 %5
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE4fillB7v160006Ev(ptr noundef %0) #1 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i32 @_ZNSt3__111char_traitsIcE3eofEv() #8
  %5 = getelementptr inbounds %"class.std::__1::basic_ios", ptr %3, i32 0, i32 2
  %6 = load i32, ptr %5, align 8
  %7 = call zeroext i1 @_ZNSt3__111char_traitsIcE11eq_int_typeEii(i32 noundef %4, i32 noundef %6) #8
  br i1 %7, label %8, label %12

8:                                                ; preds = %1
  %9 = call signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenB7v160006Ec(ptr noundef %3, i8 noundef signext 32)
  %10 = sext i8 %9 to i32
  %11 = getelementptr inbounds %"class.std::__1::basic_ios", ptr %3, i32 0, i32 2
  store i32 %10, ptr %11, align 8
  br label %12

12:                                               ; preds = %8, %1
  %13 = getelementptr inbounds %"class.std::__1::basic_ios", ptr %3, i32 0, i32 2
  %14 = load i32, ptr %13, align 8
  %15 = trunc i32 %14 to i8
  ret i8 %15
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden zeroext i1 @_ZNKSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEE6failedB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = icmp eq ptr %5, null
  ret i1 %6
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__19basic_iosIcNS_11char_traitsIcEEE8setstateB7v160006Ej(ptr noundef %0, i32 noundef %1) #1 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %5 = load ptr, ptr %3, align 8
  %6 = load i32, ptr %4, align 4
  call void @_ZNSt3__18ios_base8setstateB7v160006Ej(ptr noundef %5, i32 noundef %6)
  ret void
}

declare void @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE6sentryD1Ev(ptr noundef) unnamed_addr #3

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(ptr %0) #6 {
  %2 = call ptr @__cxa_begin_catch(ptr %0) #8
  call void @_ZSt9terminatev() #10
  unreachable
}

declare ptr @__cxa_begin_catch(ptr)

declare void @_ZSt9terminatev()

declare void @_ZNSt3__18ios_base33__set_badbit_and_consider_rethrowEv(ptr noundef) #3

declare void @__cxa_end_catch()

; Function Attrs: argmemonly nocallback nofree nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #7

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden i64 @_ZNKSt3__18ios_base5widthB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ios_base", ptr %3, i32 0, i32 3
  %5 = load i64, ptr %4, align 8
  ret i64 %5
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden i64 @_ZNSt3__115basic_streambufIcNS_11char_traitsIcEEE5sputnB7v160006EPKcl(ptr noundef %0, ptr noundef %1, i64 noundef %2) #1 align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = load i64, ptr %6, align 8
  %10 = load ptr, ptr %7, align 8
  %11 = getelementptr inbounds ptr, ptr %10, i64 12
  %12 = load ptr, ptr %11, align 8
  %13 = call i64 %12(ptr noundef %7, ptr noundef %8, i64 noundef %9)
  ret i64 %13
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC1B7v160006Emc(ptr noundef %0, i64 noundef %1, i8 noundef signext %2) unnamed_addr #1 align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca i8, align 1
  store ptr %0, ptr %4, align 8
  store i64 %1, ptr %5, align 8
  store i8 %2, ptr %6, align 1
  %7 = load ptr, ptr %4, align 8
  %8 = load i64, ptr %5, align 8
  %9 = load i8, ptr %6, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2B7v160006Emc(ptr noundef %7, i64 noundef %8, i8 noundef signext %9)
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE4dataB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE13__get_pointerB7v160006Ev(ptr noundef %3) #8
  %5 = call ptr @_ZNSt3__112__to_addressB7v160006IKcEEPT_S3_(ptr noundef %4) #8
  ret ptr %5
}

declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr noundef) unnamed_addr #3

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden i64 @_ZNSt3__18ios_base5widthB7v160006El(ptr noundef %0, i64 noundef %1) #5 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  store ptr %0, ptr %3, align 8
  store i64 %1, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = getelementptr inbounds %"class.std::__1::ios_base", ptr %6, i32 0, i32 3
  %8 = load i64, ptr %7, align 8
  store i64 %8, ptr %5, align 8
  %9 = load i64, ptr %4, align 8
  %10 = getelementptr inbounds %"class.std::__1::ios_base", ptr %6, i32 0, i32 3
  store i64 %9, ptr %10, align 8
  %11 = load i64, ptr %5, align 8
  ret i64 %11
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEC2B7v160006Emc(ptr noundef %0, i64 noundef %1, i8 noundef signext %2) unnamed_addr #1 align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca i64, align 8
  %6 = alloca i8, align 1
  %7 = alloca %"struct.std::__1::__default_init_tag", align 1
  %8 = alloca %"struct.std::__1::__default_init_tag", align 1
  store ptr %0, ptr %4, align 8
  store i64 %1, ptr %5, align 8
  store i8 %2, ptr %6, align 1
  %9 = load ptr, ptr %4, align 8
  %10 = getelementptr inbounds %"class.std::__1::basic_string", ptr %9, i32 0, i32 0
  call void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC1B7v160006INS_18__default_init_tagESA_EEOT_OT0_(ptr noundef %10, ptr noundef nonnull align 1 dereferenceable(1) %7, ptr noundef nonnull align 1 dereferenceable(1) %8)
  %11 = load i64, ptr %5, align 8
  %12 = load i8, ptr %6, align 1
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEmc(ptr noundef %9, i64 noundef %11, i8 noundef signext %12)
  call void @_ZNSt3__119__debug_db_insert_cB7v160006INS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEEEvPT_(ptr noundef %9)
  ret void
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC1B7v160006INS_18__default_init_tagESA_EEOT_OT0_(ptr noundef %0, ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2) unnamed_addr #1 align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = load ptr, ptr %6, align 8
  call void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC2B7v160006INS_18__default_init_tagESA_EEOT_OT0_(ptr noundef %7, ptr noundef nonnull align 1 dereferenceable(1) %8, ptr noundef nonnull align 1 dereferenceable(1) %9)
  ret void
}

declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEmc(ptr noundef, i64 noundef, i8 noundef signext) #3

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__119__debug_db_insert_cB7v160006INS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEEEvPT_(ptr noundef %0) #5 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  ret void
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr void @_ZNSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_EC2B7v160006INS_18__default_init_tagESA_EEOT_OT0_(ptr noundef %0, ptr noundef nonnull align 1 dereferenceable(1) %1, ptr noundef nonnull align 1 dereferenceable(1) %2) unnamed_addr #1 align 2 {
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca %"struct.std::__1::__default_init_tag", align 1
  %8 = alloca %"struct.std::__1::__default_init_tag", align 1
  store ptr %0, ptr %4, align 8
  store ptr %1, ptr %5, align 8
  store ptr %2, ptr %6, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = load ptr, ptr %5, align 8
  call void @_ZNSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EEC2B7v160006ENS_18__default_init_tagE(ptr noundef %9)
  %11 = load ptr, ptr %6, align 8
  call void @_ZNSt3__122__compressed_pair_elemINS_9allocatorIcEELi1ELb1EEC2B7v160006ENS_18__default_init_tagE(ptr noundef %9)
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EEC2B7v160006ENS_18__default_init_tagE(ptr noundef %0) unnamed_addr #5 align 2 {
  %2 = alloca %"struct.std::__1::__default_init_tag", align 1
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = getelementptr inbounds %"struct.std::__1::__compressed_pair_elem", ptr %4, i32 0, i32 0
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__122__compressed_pair_elemINS_9allocatorIcEELi1ELb1EEC2B7v160006ENS_18__default_init_tagE(ptr noundef %0) unnamed_addr #5 align 2 {
  %2 = alloca %"struct.std::__1::__default_init_tag", align 1
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  call void @_ZNSt3__19allocatorIcEC2B7v160006Ev(ptr noundef %4) #8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__19allocatorIcEC2B7v160006Ev(ptr noundef %0) unnamed_addr #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIcEEEC2B7v160006Ev(ptr noundef %3) #8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__116__non_trivial_ifILb1ENS_9allocatorIcEEEC2B7v160006Ev(ptr noundef %0) unnamed_addr #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNSt3__112__to_addressB7v160006IKcEEPT_S3_(ptr noundef %0) #5 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE13__get_pointerB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call zeroext i1 @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE9__is_longB7v160006Ev(ptr noundef %3) #8
  br i1 %4, label %5, label %7

5:                                                ; preds = %1
  %6 = call ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE18__get_long_pointerB7v160006Ev(ptr noundef %3) #8
  br label %9

7:                                                ; preds = %1
  %8 = call ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE19__get_short_pointerB7v160006Ev(ptr noundef %3) #8
  br label %9

9:                                                ; preds = %7, %5
  %10 = phi ptr [ %6, %5 ], [ %8, %7 ]
  ret ptr %10
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden zeroext i1 @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE9__is_longB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca i1, align 1
  %3 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  %4 = load ptr, ptr %3, align 8
  %5 = call zeroext i1 @_ZNSt3__130__libcpp_is_constant_evaluatedB7v160006Ev() #8
  br i1 %5, label %6, label %7

6:                                                ; preds = %1
  store i1 true, ptr %2, align 1
  br label %15

7:                                                ; preds = %1
  %8 = getelementptr inbounds %"class.std::__1::basic_string", ptr %4, i32 0, i32 0
  %9 = call noundef nonnull align 8 dereferenceable(24) ptr @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstB7v160006Ev(ptr noundef %8) #8
  %10 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__rep", ptr %9, i32 0, i32 0
  %11 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__short", ptr %10, i32 0, i32 0
  %12 = load i8, ptr %11, align 8
  %13 = and i8 %12, 1
  %14 = icmp ne i8 %13, 0
  store i1 %14, ptr %2, align 1
  br label %15

15:                                               ; preds = %7, %6
  %16 = load i1, ptr %2, align 1
  ret i1 %16
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE18__get_long_pointerB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::basic_string", ptr %3, i32 0, i32 0
  %5 = call noundef nonnull align 8 dereferenceable(24) ptr @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstB7v160006Ev(ptr noundef %4) #8
  %6 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__rep", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__long", ptr %6, i32 0, i32 2
  %8 = load ptr, ptr %7, align 8
  ret ptr %8
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNKSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE19__get_short_pointerB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::basic_string", ptr %3, i32 0, i32 0
  %5 = call noundef nonnull align 8 dereferenceable(24) ptr @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstB7v160006Ev(ptr noundef %4) #8
  %6 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__rep", ptr %5, i32 0, i32 0
  %7 = getelementptr inbounds %"struct.std::__1::basic_string<char>::__short", ptr %6, i32 0, i32 2
  %8 = getelementptr inbounds [23 x i8], ptr %7, i64 0, i64 0
  %9 = call ptr @_ZNSt3__114pointer_traitsIPKcE10pointer_toB7v160006ERS1_(ptr noundef nonnull align 1 dereferenceable(1) %8) #8
  ret ptr %9
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden zeroext i1 @_ZNSt3__130__libcpp_is_constant_evaluatedB7v160006Ev() #5 {
  ret i1 false
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden noundef nonnull align 8 dereferenceable(24) ptr @_ZNKSt3__117__compressed_pairINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repES5_E5firstB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call noundef nonnull align 8 dereferenceable(24) ptr @_ZNKSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EE5__getB7v160006Ev(ptr noundef %3) #8
  ret ptr %4
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden noundef nonnull align 8 dereferenceable(24) ptr @_ZNKSt3__122__compressed_pair_elemINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE5__repELi0ELb0EE5__getB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"struct.std::__1::__compressed_pair_elem", ptr %3, i32 0, i32 0
  ret ptr %4
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNSt3__114pointer_traitsIPKcE10pointer_toB7v160006ERS1_(ptr noundef nonnull align 1 dereferenceable(1) %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__119ostreambuf_iteratorIcNS_11char_traitsIcEEEC2B7v160006ERNS_13basic_ostreamIcS2_EE(ptr noundef %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #5 align 2 personality ptr @__gxx_personality_v0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = getelementptr inbounds %"class.std::__1::ostreambuf_iterator", ptr %7, i32 0, i32 0
  %9 = load ptr, ptr %4, align 8
  %10 = load ptr, ptr %9, align 8
  %11 = getelementptr i8, ptr %10, i64 -24
  %12 = load i64, ptr %11, align 8
  %13 = getelementptr inbounds i8, ptr %9, i64 %12
  %14 = invoke ptr @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5rdbufB7v160006Ev(ptr noundef %13)
          to label %15 unwind label %16

15:                                               ; preds = %2
  store ptr %14, ptr %8, align 8
  ret void

16:                                               ; preds = %2
  %17 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %18 = extractvalue { ptr, i32 } %17, 0
  store ptr %18, ptr %5, align 8
  %19 = extractvalue { ptr, i32 } %17, 1
  store i32 %19, ptr %6, align 4
  br label %20

20:                                               ; preds = %16
  %21 = load ptr, ptr %5, align 8
  call void @__cxa_call_unexpected(ptr %21) #9
  unreachable
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5rdbufB7v160006Ev(ptr noundef %0) #1 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_ZNKSt3__18ios_base5rdbufB7v160006Ev(ptr noundef %3)
  ret ptr %4
}

declare void @__cxa_call_unexpected(ptr)

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden ptr @_ZNKSt3__18ios_base5rdbufB7v160006Ev(ptr noundef %0) #5 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %"class.std::__1::ios_base", ptr %3, i32 0, i32 6
  %5 = load ptr, ptr %4, align 8
  ret ptr %5
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr zeroext i1 @_ZNSt3__111char_traitsIcE11eq_int_typeEii(i32 noundef %0, i32 noundef %1) #5 align 2 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = icmp eq i32 %5, %6
  ret i1 %7
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr i32 @_ZNSt3__111char_traitsIcE3eofEv() #5 align 2 {
  ret i32 -1
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden signext i8 @_ZNKSt3__19basic_iosIcNS_11char_traitsIcEEE5widenB7v160006Ec(ptr noundef %0, i8 noundef signext %1) #1 align 2 personality ptr @__gxx_personality_v0 {
  %3 = alloca ptr, align 8
  %4 = alloca i8, align 1
  %5 = alloca %"class.std::__1::locale", align 8
  %6 = alloca ptr, align 8
  %7 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i8 %1, ptr %4, align 1
  %8 = load ptr, ptr %3, align 8
  call void @_ZNKSt3__18ios_base6getlocEv(ptr sret(%"class.std::__1::locale") align 8 %5, ptr noundef %8)
  %9 = invoke noundef nonnull align 8 dereferenceable(25) ptr @_ZNSt3__19use_facetB7v160006INS_5ctypeIcEEEERKT_RKNS_6localeE(ptr noundef nonnull align 8 dereferenceable(8) %5)
          to label %10 unwind label %14

10:                                               ; preds = %2
  %11 = load i8, ptr %4, align 1
  %12 = invoke signext i8 @_ZNKSt3__15ctypeIcE5widenB7v160006Ec(ptr noundef %9, i8 noundef signext %11)
          to label %13 unwind label %14

13:                                               ; preds = %10
  call void @_ZNSt3__16localeD1Ev(ptr noundef %5)
  ret i8 %12

14:                                               ; preds = %10, %2
  %15 = landingpad { ptr, i32 }
          cleanup
  %16 = extractvalue { ptr, i32 } %15, 0
  store ptr %16, ptr %6, align 8
  %17 = extractvalue { ptr, i32 } %15, 1
  store i32 %17, ptr %7, align 4
  invoke void @_ZNSt3__16localeD1Ev(ptr noundef %5)
          to label %18 unwind label %24

18:                                               ; preds = %14
  br label %19

19:                                               ; preds = %18
  %20 = load ptr, ptr %6, align 8
  %21 = load i32, ptr %7, align 4
  %22 = insertvalue { ptr, i32 } undef, ptr %20, 0
  %23 = insertvalue { ptr, i32 } %22, i32 %21, 1
  resume { ptr, i32 } %23

24:                                               ; preds = %14
  %25 = landingpad { ptr, i32 }
          catch ptr null
  %26 = extractvalue { ptr, i32 } %25, 0
  call void @__clang_call_terminate(ptr %26) #10
  unreachable
}

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden noundef nonnull align 8 dereferenceable(25) ptr @_ZNSt3__19use_facetB7v160006INS_5ctypeIcEEEERKT_RKNS_6localeE(ptr noundef nonnull align 8 dereferenceable(8) %0) #1 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_ZNKSt3__16locale9use_facetERNS0_2idE(ptr noundef %3, ptr noundef nonnull align 8 dereferenceable(12) @_ZNSt3__15ctypeIcE2idE)
  ret ptr %4
}

declare void @_ZNKSt3__18ios_base6getlocEv(ptr sret(%"class.std::__1::locale") align 8, ptr noundef) #3

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden signext i8 @_ZNKSt3__15ctypeIcE5widenB7v160006Ec(ptr noundef %0, i8 noundef signext %1) #1 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i8, align 1
  store ptr %0, ptr %3, align 8
  store i8 %1, ptr %4, align 1
  %5 = load ptr, ptr %3, align 8
  %6 = load i8, ptr %4, align 1
  %7 = load ptr, ptr %5, align 8
  %8 = getelementptr inbounds ptr, ptr %7, i64 7
  %9 = load ptr, ptr %8, align 8
  %10 = call signext i8 %9(ptr noundef %5, i8 noundef signext %6)
  ret i8 %10
}

declare void @_ZNSt3__16localeD1Ev(ptr noundef) unnamed_addr #3

declare ptr @_ZNKSt3__16locale9use_facetERNS0_2idE(ptr noundef, ptr noundef nonnull align 8 dereferenceable(12)) #3

; Function Attrs: noinline optnone ssp uwtable
define linkonce_odr hidden void @_ZNSt3__18ios_base8setstateB7v160006Ej(ptr noundef %0, i32 noundef %1) #1 align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  store ptr %0, ptr %3, align 8
  store i32 %1, ptr %4, align 4
  %5 = load ptr, ptr %3, align 8
  %6 = getelementptr inbounds %"class.std::__1::ios_base", ptr %5, i32 0, i32 4
  %7 = load i32, ptr %6, align 8
  %8 = load i32, ptr %4, align 4
  %9 = or i32 %7, %8
  call void @_ZNSt3__18ios_base5clearEj(ptr noundef %5, i32 noundef %9)
  ret void
}

declare void @_ZNSt3__18ios_base5clearEj(ptr noundef, i32 noundef) #3

; Function Attrs: noinline nounwind optnone ssp uwtable
define linkonce_odr hidden i64 @_ZNSt3__118__constexpr_strlenB7v160006EPKc(ptr noundef %0) #5 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i64 @strlen(ptr noundef %3) #8
  ret i64 %4
}

; Function Attrs: nounwind
declare i64 @strlen(ptr noundef) #4

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE3putEc(ptr noundef, i8 noundef signext) #3

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSt3__113basic_ostreamIcNS_11char_traitsIcEEE5flushEv(ptr noundef) #3

declare i32 @__gxx_personality_v0(...)

attributes #0 = { "objc_arc_inert" }
attributes #1 = { noinline optnone ssp uwtable "darwin-stkchk-strong-link" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "probe-stack"="___chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #2 = { nonlazybind }
attributes #3 = { "darwin-stkchk-strong-link" "frame-pointer"="all" "no-trapping-math"="true" "probe-stack"="___chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind "darwin-stkchk-strong-link" "frame-pointer"="all" "no-trapping-math"="true" "probe-stack"="___chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #5 = { noinline nounwind optnone ssp uwtable "darwin-stkchk-strong-link" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "probe-stack"="___chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" }
attributes #6 = { noinline noreturn nounwind }
attributes #7 = { argmemonly nocallback nofree nounwind willreturn }
attributes #8 = { nounwind }
attributes #9 = { noreturn }
attributes #10 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 2]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!5 = !{i32 1, !"Objective-C Class Properties", i32 64}
!6 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i8 0}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 8, !"PIC Level", i32 2}
!9 = !{i32 7, !"uwtable", i32 2}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"Apple clang version 15.0.0 (clang-1500.1.0.2.5)"}
!12 = !{}
