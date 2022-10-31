# frida实践

​    

# 一、frida python实践

## 1、get_device_manager

获取设备管理器

```python
manager = frida.get_device_manager()
```

   

## 2、enumerate_devices

选择设备连接方式

```python
# a. 通过 device manager 管理器获取设备对象
devices = manager.enumerate_devices()
> Device(id="local", name="Local System", type='local')
> Device(id="socket", name="Local Socket", type='remote')
> Device(id="b868ca03", name="M2011J18C", type='usb')
device = manager.get_device(devices[2].id, timeout)  # 指定设备id连接

# b. 指定设备连接方式
device = frida.get_local_device()      # Local System
device = frida.get_usb_device(timeout) # usb
device = frida.get_remote_device()     # 远程, Local Socket
```

​    

## 3、enumerate_applications 和enumerate_processes

获取device上的所有App和进程

```python
# 等价于 frida-ps -aU
#  PID  Name    Identifier
#  5  ------  ---------------
#  21550  Gadget  re.frida.Gadget

pid = None
for a in device.enumerate_applications():
    if a.identifier == 're.frida.Gadget':
        pid = a.pid
        break

## 补充：获取设备上正则运行的进程信息
# 等价于 frida-ps -a
#  PID  Name
#  -----  ------
#  32429  Gadget

all_processes = device.enumerate_processes()
    for per_process in all_processes:
        print(per_process)
```

​    

## 4、attach

连接设备, 获取session

```python
# 等价于 frida -U -f xx.xx.xx --no-pause
# 启动应用进入交互模式, 应用于 App 未打开的情景

# a. pid 方式, app package (identifier)，适用于iOS
pid = device.spawn([self.package])
device.resume(pid)
time.sleep(2)  # Without it Java.perform silently fails
session = device.attach(pid)

# b.app name，适用于Android
subprocess.call("adb shell pm clear app.identifier", shell=True)         # 清理APP应用数据
subprocess.call("adb shell monkey -p app.identifier -v 1", shell=True)  # 通过monkey指令指定包名唤醒APP
time.sleep(5)
session = device.attach('app name')
# session = device.attach('app identifier')
```

​    

## 5、create_script

注入JS脚本

```python
"""
def create_script(
        self, source: str, name: Optional[str] = None, snapshot: Optional[bytes] = None, runtime: Optional[str] = None
    ) -> Script
"""

# a. js字符串
jsScript = """
    console.log('this is inject javascript code')
"""
script = session.create_script(jsScript)

# b. js文件
with open("hook.js",mode='r',encoding='UTF-8') as f:
    Log.info('Inject script name: ' + full_js_file_name)
    script = session.create_script(f.read())
```

```javascript
// 程序入口: hook.js

Java.perform(function() 
{
    // 获取类
    var clazz = Java.use("com.unity3d.player.UnityPlayerActivity");
    // 获取类中所有函数
    var methods = clazz.class.getDeclaredMethods();

    console.log("have method count:"+methods.length);
	var i=0
    if(methods.length > 0){
        //遍历函数名
        methods.forEach(function(method){
			i = i+1
            console.log(i+":"+method);
        });
    }

});
```

   

## 6、load

打印日志&执行注入

```python
# 打印js注入信息
def on_message(message, data):
    if message['type'] == 'send':
        Log.send(message['payload'])
    elif message['type'] == 'error':
        Log.error(message['description'])
    else:
        Log.error(message)

script.on("message", on_message)

# 执行
script.load()

# prevent the python script from terminating
log.info('Waiting for JavaScript...')
sys.stdin.read()
```

​      

## 7、Demo示例



### 1）初始化设备

```python
import frida

# 初始化设备连接
def init_device():
    Log.info('Current frida version: ' + str(frida.__version__))
    
    # 获取设备管理器
    manager = frida.get_device_manager()
    Log.print('Select a frida device:')

    # 默认设备连接方式
    devices = manager.enumerate_devices()
    for i, ldevice in enumerate(devices, 1):
        Log.print(str(i) + ' => ' + str(ldevice))

    # 选择设备连接方式
    select = int(input())
    if select > len(devices):
        Log.error('Out of range.')
        sys.exit(1)

    device_id = devices[select - 1].id
	
    # 链接设备: 获取指定 UID 设备
    device = manager.get_device(device_id, 1)
    Log.info('Connect to device \'' + device.name + '\' successfully.')

    return device


if __name__ == '__main__':
    try:
        device = init_device()

        # 遍历需要hook的APP&进程列表
        for per_hook_process in processes_to_hook:
            # 链接设备
            session = attach_android(per_hook_process['name'], per_hook_process['identifier'])

            # js 脚本注入
            for js_module in js_modules:
                process_name_var = 'var __process_name = "' + per_hook_process['identifier'] + '";'
                module_name_var = 'var __module_name = "' + js_module['name'] + '";'
                full_js_file_name = 'example/hook_' + js_module['type'] + '_' + js_module['name'] + '.js'

                with open(full_js_file_name) as f:
                    Log.info('Inject script name: ' + full_js_file_name)
                    script = session.create_script(process_name_var + module_name_var + f.read())

                script.on('message', on_message)
                Log.info('Load script name: ' + full_js_file_name)
                script.load()

        Log.info('Waiting for JavaScript...')
        print('----------------------------------------')
        sys.stdin.read()

    except Exception as e:
        Log.error(repr(e))
```

​     

### 2）设备连接 Android & iOS

#### a. Android

```python
def attach_android(app_name: str, app_identifier: str):
    """Android设备连接方式
    """
    try:
        # 清理APP应用数据
        Log.info('Launching process \'' + app_name + '\'')
        subprocess.call("adb shell pm clear " + app_identifier, shell=True)

        # 通过monkey指令指定包名唤醒APP
        subprocess.call("adb shell monkey -v 1 -p " + app_identifier, shell=True)
    except frida.ExecutableNotFoundError as e2:
        Log.error('Unable to find execuable \'' + app_name + '\'.')

    Log.info('Attaching process \'' + app_name + '\'')
    time.sleep(5)

    return device.attach(app_name)
```

#### b. iOS

```python
def attach_ios(app_name: str, app_identifier: str):
    """iOS设备连接方式
    """
    try:
        device.get_process(app_name)
    except frida.ProcessNotFoundError as e:
        Log.warn('Unable to find process \'' + app_name + '\', try to spawn...')

        # Must use identifier to spawn
        try:
            pid = device.spawn(app_identifier)
            device.resume(pid)
            time.sleep(5)
        except frida.ExecutableNotFoundError as e2:
            Log.error('Unable to find execuable \'' + app_name + '\'.')

    Log.info('Attaching: ' + app_name)

    return device.attach(app_name)
```

   

### 3）js脚本注入

```python
js_modules = [
    {'type': 'android', 'name': 'env'},
]

def inject_js(session: frida.core.Session, modules: list, app_identifier: str):
    """注入JS脚本
    """
    # js 脚本注入
    for js_module in modules:
        process_name_var = 'var __process_name = "' + app_identifier + '";'
        module_name_var = 'var __module_name = "' + js_module['name'] + '";'
        full_js_file_name = 'example/hook_' + js_module['type'] + '_' + js_module['name'] + '.js'
		
        # 加载js脚本
        with open(full_js_file_name) as f:
            Log.info('Inject script name: ' + full_js_file_name)
            script = session.create_script(process_name_var + module_name_var + f.read())
		
        # 打印日志
        script.on('message', on_message)
        
        # 执行注入
        Log.info('Load script name: ' + full_js_file_name)
        script.load()
```

   

### 4）执行

```python
if __name__ == '__main__':
    try:
        device = init_device()

        # 遍历需要hook的APP&进程列表
        for per_hook_process in processes_to_hook:
            # 链接设备
            session = attach_android(per_hook_process['name'], per_hook_process['identifier'])

            # js 脚本注入
            inject_js(session, js_modules)

        Log.info('Waiting for JavaScript...')
        print('----------------------------------------')
        sys.stdin.read()

    except Exception as e:
        Log.error(repr(e))
```

​    

# 二、frida注入案例

​     





# 附录

1. [FRIDA 实用手册](https://zhuanlan.zhihu.com/p/56702055)
2. [hacktricks: frida-tutorial-2](https://book.hacktricks.xyz/mobile-pentesting/android-app-pentesting/frida-tutorial/frida-tutorial-2)
3. [Python frida.get_device() Examples](https://www.programcreek.com/python/example/111316/frida.get_device)
4. [Frida从入门到放弃](http://www.gouzai.pw/2019/03/07/Frida%E4%BB%8E%E5%85%A5%E9%97%A8%E5%88%B0%E6%94%BE%E5%BC%83/)