cmake_minimum_required(VERSION 2.8)

PROJECT( SupResTrainer )

set(ITK_DIR "/home/rifat/Apps/ITK/InsightToolkit-4.2.0b/")
set(VTK_DIR "/home/rifat/Apps/VTK5_10/VTKb/")

FIND_PACKAGE ( ITK REQUIRED)
IF (  ITK_FOUND  )
         INCLUDE(  ${ITK_USE_FILE}  )
ENDIF(   ITK_FOUND   )

FIND_PACKAGE ( VTK REQUIRED)
IF (   VTK_FOUND   )
         INCLUDE(  ${VTK_USE_FILE}  )
ENDIF(   VTK_FOUND   )

include_directories ("${PROJECT_SOURCE_DIR}/FeatureExtractor")
add_subdirectory (FeatureExtractor) 

ADD_EXECUTABLE( SupResTrainer   SuperResolutionTrainer.cxx)

TARGET_LINK_LIBRARIES ( SupResTrainer FeatureExtractor ${ITK_LIBRARIES} ${VTK_LIBRARIES})
