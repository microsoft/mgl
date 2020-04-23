set(TARGET mgl_pairhmm)

find_package(TBB CONFIG REQUIRED)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

set_property(SOURCE compute_prob_avxd.cc APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx -mavx2 -mfma ")
set_property(SOURCE compute_prob_avxf.cc APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx -mavx2 -mfma ")

add_library(${TARGET} SHARED 
    com_microsoft_mgl_pairhmm_MicrosoftPairHmm.cc
    pairhmm_common.cc compute_prob_scalar.cc compute_prob_avxd.cc compute_prob_avxf.cc compute_fast_prob.cc)

target_link_libraries(mgl_pairhmm PRIVATE TBB::tbb)
target_link_libraries(mgl_pairhmm PRIVATE TBB::tbbmalloc)

install(TARGETS ${TARGET} DESTINATION ${CMAKE_BINARY_DIR})
