# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chris/projects/feature_compare

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chris/projects/feature_compare/build

# Include any dependencies generated for this target.
include CMakeFiles/orb_flann.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/orb_flann.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/orb_flann.dir/flags.make

CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o: CMakeFiles/orb_flann.dir/flags.make
CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o: ../ORB_FlannMatcher.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/chris/projects/feature_compare/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o -c /home/chris/projects/feature_compare/ORB_FlannMatcher.cpp

CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/chris/projects/feature_compare/ORB_FlannMatcher.cpp > CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.i

CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/chris/projects/feature_compare/ORB_FlannMatcher.cpp -o CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.s

CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.requires:
.PHONY : CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.requires

CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.provides: CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.requires
	$(MAKE) -f CMakeFiles/orb_flann.dir/build.make CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.provides.build
.PHONY : CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.provides

CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.provides.build: CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o

# Object files for target orb_flann
orb_flann_OBJECTS = \
"CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o"

# External object files for target orb_flann
orb_flann_EXTERNAL_OBJECTS =

orb_flann: CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o
orb_flann: CMakeFiles/orb_flann.dir/build.make
orb_flann: /usr/local/lib/libopencv_viz.so.2.4.9
orb_flann: /usr/local/lib/libopencv_videostab.so.2.4.9
orb_flann: /usr/local/lib/libopencv_video.so.2.4.9
orb_flann: /usr/local/lib/libopencv_ts.a
orb_flann: /usr/local/lib/libopencv_superres.so.2.4.9
orb_flann: /usr/local/lib/libopencv_stitching.so.2.4.9
orb_flann: /usr/local/lib/libopencv_photo.so.2.4.9
orb_flann: /usr/local/lib/libopencv_ocl.so.2.4.9
orb_flann: /usr/local/lib/libopencv_objdetect.so.2.4.9
orb_flann: /usr/local/lib/libopencv_nonfree.so.2.4.9
orb_flann: /usr/local/lib/libopencv_ml.so.2.4.9
orb_flann: /usr/local/lib/libopencv_legacy.so.2.4.9
orb_flann: /usr/local/lib/libopencv_imgproc.so.2.4.9
orb_flann: /usr/local/lib/libopencv_highgui.so.2.4.9
orb_flann: /usr/local/lib/libopencv_gpu.so.2.4.9
orb_flann: /usr/local/lib/libopencv_flann.so.2.4.9
orb_flann: /usr/local/lib/libopencv_features2d.so.2.4.9
orb_flann: /usr/local/lib/libopencv_core.so.2.4.9
orb_flann: /usr/local/lib/libopencv_contrib.so.2.4.9
orb_flann: /usr/local/lib/libopencv_calib3d.so.2.4.9
orb_flann: /usr/lib/x86_64-linux-gnu/libGLU.so
orb_flann: /usr/lib/x86_64-linux-gnu/libGL.so
orb_flann: /usr/lib/x86_64-linux-gnu/libSM.so
orb_flann: /usr/lib/x86_64-linux-gnu/libICE.so
orb_flann: /usr/lib/x86_64-linux-gnu/libX11.so
orb_flann: /usr/lib/x86_64-linux-gnu/libXext.so
orb_flann: /usr/local/lib/libopencv_nonfree.so.2.4.9
orb_flann: /usr/local/lib/libopencv_ocl.so.2.4.9
orb_flann: /usr/local/lib/libopencv_gpu.so.2.4.9
orb_flann: /usr/local/lib/libopencv_photo.so.2.4.9
orb_flann: /usr/local/lib/libopencv_objdetect.so.2.4.9
orb_flann: /usr/local/lib/libopencv_legacy.so.2.4.9
orb_flann: /usr/local/lib/libopencv_video.so.2.4.9
orb_flann: /usr/local/lib/libopencv_ml.so.2.4.9
orb_flann: /usr/local/lib/libopencv_calib3d.so.2.4.9
orb_flann: /usr/local/lib/libopencv_features2d.so.2.4.9
orb_flann: /usr/local/lib/libopencv_highgui.so.2.4.9
orb_flann: /usr/local/lib/libopencv_imgproc.so.2.4.9
orb_flann: /usr/local/lib/libopencv_flann.so.2.4.9
orb_flann: /usr/local/lib/libopencv_core.so.2.4.9
orb_flann: CMakeFiles/orb_flann.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable orb_flann"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/orb_flann.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/orb_flann.dir/build: orb_flann
.PHONY : CMakeFiles/orb_flann.dir/build

CMakeFiles/orb_flann.dir/requires: CMakeFiles/orb_flann.dir/ORB_FlannMatcher.cpp.o.requires
.PHONY : CMakeFiles/orb_flann.dir/requires

CMakeFiles/orb_flann.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/orb_flann.dir/cmake_clean.cmake
.PHONY : CMakeFiles/orb_flann.dir/clean

CMakeFiles/orb_flann.dir/depend:
	cd /home/chris/projects/feature_compare/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chris/projects/feature_compare /home/chris/projects/feature_compare /home/chris/projects/feature_compare/build /home/chris/projects/feature_compare/build /home/chris/projects/feature_compare/build/CMakeFiles/orb_flann.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/orb_flann.dir/depend
