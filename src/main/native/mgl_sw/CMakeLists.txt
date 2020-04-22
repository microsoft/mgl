set(TARGET mgl_sw)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")

set_property(SOURCE sw_avx.cpp APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx -mavx2 ")

add_library(${TARGET} SHARED
    com_microsoft_mgl_smithwaterman_MicrosoftSmithWaterman.cpp cpuid.cpp sw.cpp sw_avx.cpp)

install(TARGETS ${TARGET} DESTINATION ${CMAKE_BINARY_DIR})

