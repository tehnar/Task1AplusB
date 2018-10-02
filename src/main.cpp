#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


cl_device_id getBestDevice() {
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    if (!platformsCount) {
        std::cout << "No platforms found" << std::endl;
        return nullptr;
    }

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    cl_device_id cpuDeviceId = nullptr;

    for (cl_platform_id platformId: platforms) {
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id > devices(platformsCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (cl_device_id deviceId: devices) {
            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr));
            if (deviceType & CL_DEVICE_TYPE_GPU) {
                return deviceId;
            }
            if (deviceType & CL_DEVICE_TYPE_CPU) {
                cpuDeviceId = deviceId;
            }

        }
    }

    return cpuDeviceId;
}

cl_platform_id getPlatformForDevice(cl_device_id deviceId) {
    cl_platform_id platformId;
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platformId, nullptr));
    return platformId;
}

cl_context createContext(cl_device_id deviceId) {
    cl_platform_id platformId = getPlatformForDevice(deviceId);

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (intptr_t)(platformId), 0};
    cl_int contextErrorCode = 0;
    cl_context context = clCreateContext(
            properties, 1, std::vector<cl_device_id> {deviceId}.data(), nullptr, nullptr, &contextErrorCode);
    OCL_SAFE_CALL(contextErrorCode);
    return context;
}

cl_command_queue createCommandQueue(cl_context context, cl_device_id deviceId) {
    cl_int commandQueueError = 0;
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceId, 0, &commandQueueError);
    OCL_SAFE_CALL(commandQueueError);
    return commandQueue;
}

int main()
{
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с заданием Example0EnumDevices узнайте какие есть устройства, и выберите из них какое-нибудь
    // (если есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)

    cl_device_id deviceId = getBestDevice();
    if (!deviceId) {
        std::cout << "No devices found" << std::endl;
        return 0;
    }

    cl_context context = createContext(deviceId);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue

    cl_command_queue commandQueue = createCommandQueue(context, deviceId);
    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт что чисел в каждом массиве - n штук
    // Данные в as и bs можно прогрузить этим же методом скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    cl_int bufferErrorCode = 0;
    cl_mem asBuffer = clCreateBuffer(
            context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, as.data(), &bufferErrorCode);
    OCL_SAFE_CALL(bufferErrorCode);

    cl_mem bsBuffer = clCreateBuffer(
            context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, bs.data(), &bufferErrorCode);
    OCL_SAFE_CALL(bufferErrorCode);

    cl_mem csBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n, bs.data(), &bufferErrorCode);
    OCL_SAFE_CALL(bufferErrorCode);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания)
    // напечатав исходники в консоль (if проверяет что удалось считать хоть что-то)
    std::string kernelSources;
    std::ifstream file("src/cl/aplusb.cl");
    kernelSources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (kernelSources.empty()) {
        throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
    }
//    std::cout << kernelSources << std::endl;

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание что передать вам нужно указатель на указатель
    cl_int programErrorCode = 0;
    const char *sources = kernelSources.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &sources, nullptr, &programErrorCode);
    OCL_SAFE_CALL(programErrorCode);
    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    cl_int buildErrorCode = clBuildProgram(
            program, 1, std::vector<cl_device_id>{deviceId}.data(), nullptr, nullptr, nullptr);

    // А так же напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // см. clGetProgramBuildInfo
    size_t logSize = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));
    std::vector<char> log(logSize, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr));

    if (logSize > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }
    OCL_SAFE_CALL(buildErrorCode);

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_int kernelErrorCode = 0;
    cl_kernel kernel = clCreateKernel(program, "aplusb", &kernelErrorCode);
    OCL_SAFE_CALL(kernelErrorCode);

    {
         unsigned int i = 0;
         clSetKernelArg(kernel, i++, sizeof(asBuffer), &asBuffer);
         clSetKernelArg(kernel, i++, sizeof(bsBuffer), &bsBuffer);
         clSetKernelArg(kernel, i++, sizeof(csBuffer), &csBuffer);
         clSetKernelArg(kernel, i, sizeof(n), &n);
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание что чтобы дождаться окончания вычислений (чтобы знать когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(
                    commandQueue, kernel, 1, nullptr, &globalWorkSize, &workGroupSize, 0, nullptr, &event));
            clWaitForEvents(1, &event);
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считаются не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклониение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще) достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        
        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти т.о. 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3 * n * sizeof(float) / t.lapAvg() / (1<<30) << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(
                    commandQueue, csBuffer, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / t.lapAvg() / (1<<30) << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    clReleaseKernel(kernel);

    clReleaseProgram(program);

    clReleaseMemObject(csBuffer);
    clReleaseMemObject(bsBuffer);
    clReleaseMemObject(asBuffer);

    clReleaseCommandQueue(commandQueue);

    clReleaseContext(context);

    return 0;
}
