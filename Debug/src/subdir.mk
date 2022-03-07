################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp \
../src/object_detection.cpp \
../src/read_data.cpp \
../src/write_data.cpp 

OBJS += \
./src/main.o \
./src/object_detection.o \
./src/read_data.o \
./src/write_data.o 

CPP_DEPS += \
./src/main.d \
./src/object_detection.d \
./src/read_data.d \
./src/write_data.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/ -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


