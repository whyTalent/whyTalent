# Frida框架基础内容

​     

# 一 基础概念

`frida` 是一款基于 `python` + `javascript` 的 hook 框架，可将自己编写的JavaScript代码注入到应用中，并可运行在 `android`、`ios`、`linux`、`win`等各个平台，主要使用的动态二进制插桩技术。

> **插桩技术**：指将额外的代码注入程序中以收集运行时的信息，可分为源代码插桩 SCI 和二进制插桩 BI
>
> * **源代码插桩 SCI**：Source Code Instrumentation，额外代码注入到程序源代码中
> * **二进制插桩 BI**：Binary Instrumentation，额外代码注入到二进制可执行文件中
>   * 静态二进制插桩（SBI, Static Binary Instrumentation），在程序执行前插入额外的代码和数据，生成一个永久改变的可执行文件；
>   * 动态二进制插桩（DBI, Dynamic Binary Instrumentation），在程序运行时实时插入额外代码和数据，对可执行文件没有任何永久改变。

​    

Frida功能不仅仅是 `Hook`，还包括以下这些功能：

> 访问进程的内存
>
> 在应用程序运行时覆盖一些功能
>
> 从导入的类中调用函数
>
> 在堆上查找对象实例并使用这些对象实例
>
> Hook，跟踪和拦截函数等等

​     

# 二 环境配置

​         

## 1 Frida安装

安装 frida CLI（command-line interface，命令行界面）

```shell
# python 安装(建议连接VPN)
pip install frida
pip install frida-tools

# 源码编译安装
git clone git://github.com/frida/frida.git
cd frida
make

# 注意安装版本，系列工具版本号要一致
frida --version
>> 16.0.2

# 查看当前系统进程
frida-ps
>>
  PID  Name
-----  ------------------------------------------------------------------------------
  453   AirPlayUIAgent
  471   AppSSOAgent
  491   CoreLocationAgent
  482   CoreServicesUIAgent
  ... ...
>>
```

​    

## 2 nvm安装

```shell
安装：
1、命令行输入：brew install nvm  #安装nvm
	1.1、修改 ~/.zshrc 中环境变量
	1.2、按照nvm安装后提示的变量内容进行更新 
	1.3、命令行输入：source ~/.zshrc
2、命令行输入：nvm ls-remote #查看可安装node版本
3、命令行输入：nvm install v12.13.0 #安装node12.13.0版本
	nvm常用命令：
		nvm uninstall [version] #卸载指定版本
		nvm use [--silent] [version] #切换到指定版本
		nvm ls #查看已安装版本
```

​    

## 3 设备端配置

下面主要分别介绍在 `Android` / `iOS` 两端的设备环境配置

​       

### 1) Android端

-----

电脑 USB 连接安卓手机，针对设备是否 root 采用不同的方式

#### a. root设备

>**1）**查看手机型号，下载系统对应版本的 [frida-server](https://github.com/frida/frida/releases)
>
>```shell
>$ adb shell getprop ro.product.cpu.abi
>```
>
>注意：Frida-server的版本必须跟宿主机的Frida版本一致，比如宿主机Frida的版本是10.6.52，Android手机是arm64的，那么应该下载：rida-server-10.6.52-android-arm64.xz 文件。
>
>**2）**下载后解压文件，并将文件重命名为: `frida-server`, 重命名完成后使用`adb push`命令推送到手机中
>
>```shell
>$ adb root # might be required
>$ adb push frida-server /data/lcoal/tmp
>```
>
>**3）**推送完成后将frida-sever赋予执行的权限，并运行Frida-server，使用以下命令：
>
>```shell
>$ adb shell "chmod 755 /data/local/tmp/frida-server"
>$ adb shell "/data/local/tmp/frida-server &"
>```
>
>**注1**： 如果frida-server没有启动，查看一下你是否使用的是Root用户来启动，如果使用Root用户则应该是`#`，
>
>**注2**： 如果要启动frida-server作为后台进程、可以使用这个命令`./frida-server &`
>
>**4）**正常启动后，另开一个终端，使用 `frida-ps -U` 命令检查Frida是否正常运行，如果正常运行则会列出Android设备上当前正在运行的进程。
>
>参数-U 代表USB，意思让Frida检查USB设备，使用`frida-ps -R` 也可以，但是需要进行转发，执行 `adb forward tcp:27042 tcp:27042` 修改端口号，后执行`frida-ps -R`也可以看到手机上的进程。
>
>```shell
># 进行端口转发监听
>$ adb forward tcp:27042 tcp:27042
>$ adb forward tcp:27043 tcp:27043
>
># 注：27042 用于与frida-server通信的默认端口号, 之后的每个端口对应每个注入的进程，检查27042端口可检测 Frida 是否存在
>```

​       

#### b. 非root设备

没有 root 的设备采用安装 `frida-gadget` 的方式，需要对目标应用 apk 进行反编译注入和调用

> 1）**反编译 apk**，反编译之后生成 target_app_floder 文件夹
>
> ```shell
> $ apktool d target_app.apk -o target_app_floder
> ```
>
> 2）**下载系统对应版本的 [frida-gadget](https://github.com/frida/frida/releases)**，解压并放到指定位置
>
> 下载之后将其进行解压，然后放到 `target_app_floder//lib/armeabi/libfrida-gadget.so`，注意修改名字以 `lib` 开头 `.so` 结尾，对应下一步的代码中的`frida-gadger`
>
> > **注**：测试设备是 `arm64-v8a`，所以下载 [**frida-gadget-12.2.27-android-arm64.so.xz**](https://github.com/frida/frida/releases/download/12.2.27/frida-gadget-12.2.27-android-arm64.so.xz)，但最后回编译打包之后，运行总是奔溃，不断的尝试之后才发现使用 [**frida-gadget-12.2.27-android-arm.so.xz**](https://github.com/frida/frida/releases/download/12.2.27/frida-gadget-12.2.27-android-arm.so.xz) 可以正常运行
>
> 3）**代码中加载上一步so 文件，建议在应用的入口文件中执行**
>
> 根据 AndroidManifest.xml 文件找到程序的入口文件，例如 MainActivity，在反编译生成的代码 smali 中的 onCreate 方法中注入如下代码
>
> ```java
> const-string v0, "frida-gadget"
> invoke-static {v0}, Ljava/lang/System;>loadLibrary(Ljava/lang/String;)V
> ```
>
> 4）**检查AndroidManifest.xml清单文件的网络权限**，忌重复添加，会导致回编译包出错
>
> ```java
> <uses-permission android:name="android.permission.INTERNET" />
> ```
>
> 5）**回编译 apk**
>
> > a. 重新打包
> >
> > ```shell
> > $ apktool b -o repackage.apk target_app_floder
> > ```
> >
> > b. 创建签名文件，有的话可忽略此步骤
> >
> > ```shell
> > $ keytool -genkey -v -keystore mykey.keystore -alias mykeyaliasname -keyalg RSA -keysize 2048 -validity 10000
> > ```
> >
> > c. 签名，以下任选其一
> >
> > ```shell
> > # jarsigner 方式
> > $ jarsigner -sigalg SHA256withRSA -digestalg SHA1 -keystore mykey.keystore -storepass 你的密码 repackaged.apk mykeyaliasname
> > 
> > # apksigner 方式: 如需要禁用 v2签名 添加选项--v2-signing-enabled false
> > $ apksigner sign --ks mykey.keystore --ks-key-alias mykeyaliasname repackaged.apk
> > ```
> >
> > d. 验证，以下任选其一
> >
> > ```shell
> > # jarsigner方式
> > $ jarsigner -verify repackaged.apk
> > 
> > # apksigner 方式
> > $ apksigner verify -v --print-certs repackaged.apk
> > 
> > # keytool方式
> > $ keytool -printcert -jarfile repackaged.apk
> > ```
> >
> > e. 对齐
> >
> > ```shell
> > # 4字节对齐优化
> > $ zipalign -v 4 repackaged.apk final.apk
> > 
> > # 检查是否对齐
> > $ zipalign -c -v 4 final.apk
> > 
> > # zipalign可以在V1签名后执行, 但zipalign不能在V2签名后执行, 只能在V2签名之前执行
> > ```
>
> 6）**安装 apk**
>
> ```shell
> $ adb install final.apk
> ```
>
> 7）**检查是否成功**
>
> 打开运行 final.apk，在注入代码位置进入停止等待页面

​     

**另一种非 root 方式**：https://bbs.pediy.com/thread-229970.htm

​         

### 2) iOS端

-----

在iOS设备上，Frida支持两种使用模式，具体使用哪种模式要看你的iOS设备是否已经越狱

#### a. 已越狱设备

越狱机上使用Cydia工具配置Frida

> 1）启动 Cydia
>
> 2）添加软件源：manage -> 软件源 Sources-> 编辑 Edit（左上角）-> 添加 Add（右上角）-> 输入 https://build.frida.re/
>
> 3）通过刚才添加的软件源安装 frida 插件，注意需要根据手机进行安装：iPhone 5 及之前的机器为 32 位，5s 及之后的机器为 64 位，进入 变更 -> 找到Frida -> 进入Frida 在右上角点击安装

​    

#### b. 未越狱设备

frida-server在运行时需要root环境，但如果没有越狱的设备，依然可以使用frida，只需要重打包ipa文件，将frida运行库注入ipa文件中，app在启动时会自动加载frida运行库，即可实现在非越狱的设备上使用Frida。

因此，为了让一个App能使用Frida，必须想办法让它加载一个 **.dylib**，就是一个 **Gadget** 模块，因此需要配置一下 **xcode** 的编译配置来让你的App可以集成Frida。当然也可以使用相关的工具来修改一个已经编译好的App， 比如 **insert_dylib** 这样的工具。

... ...

​      

#### c. 模拟器

在模拟器中进行测试，需要把命令行中的 **-U** 替换成 **-R**，这样一来底层的内部调用也从 **get_usb_device()** 变成 **get_remote_device()**

... ...

​      

# 三 能力

​       

## 1 Frida 基础工具

Frida 支持一下基础工具：

### 1）frida-ls-devices

查看可用的设备列表

   

### 2）frida-ps

获取设备进程列表信息

<div align="center"><img src="
  imgs/frida-ps.png" alt="frida-ps" style="zoom:80%;" /></div>

```shell
# usage: frida-ps [options]

# 连接到 USB 设备查看进程列表
frida-ps -U

# 连接到 USB 设备查看正在运行的应用
frida-ps -U -a

# 连接到 USB 设备查看所有安装的应用
frida-ps -U -a -i

# 连接到指定的 USB 设备查看正在运行的应用
frida-ps -D xxxxxx -a
```

​     

### 3）frida-kill

结束/杀死设备上的指定进程

```shell
# usage: frida-kill [options] process
# 比如: 杀掉 PID 为 26745 的 Twitter
frida-kill -U 26745
frida-kill -U Twitter
frida-kill -D xxxxxxxxx 26745
frida-kill -D xxxxxxxxx Twitter
```

​     

### 4）frida-trace

跟踪函数或方法的调用

```shell
# usage: frida-trace [options] target

# a. 跟踪函数调用
#   -i: 包含某个函数，支持模糊匹配
#		-x: 排除某个函数，支持模糊匹配

# ps: 跟踪名为 compress 和以 recv 开头的函数，且排除以 recvmsg 开头的函数; 明朗在目标 App 打开后执行的
frida-trace -U -i compress -i "recv*" -x "recvmsg*" Twitter
# ps: 强制启动 App 来进行跟踪，可以使用 -f 应用的BundleID 参数
frida-trace -U -i compress -i "recv*" -x "recvmsg*" -f "com.atebits.Tweetie2"

# b. 跟踪 OC 方法的调用
#		-m: 包含某个方法，支持模糊匹配
#		-M: 排除某个方法，支持模糊匹配
frida-trace -U -m "-[T1HomeTimelineItemsViewController _load*]" -M "-[T1HomeTimelineItemsViewController _loadBottomWithSource:]" Twitter

# c. 跟踪调用栈: 只需要在 JS 文件中添加如下代码片段即可跟踪某个方法的调用栈
console.log('\tBacktrace:\n\t' + Thread.backtrace(this.context, Backtracer.ACCURATE).map(DebugSymbol.fromAddress).join('\n\t'));
```

​    

### 5）frida CLI

交互模式

```shell
# 提供了两种进入交互模式的方式
# 参数
# -l 指定加载一个Javascript脚本
# -U 指定对USB设备操作
# frida运行过程中，执行%resume重新注入，执行%reload来重新加载脚本；执行exit结束脚本注入

# a. 通过应用名或 PID 附加, 应用于 App 已打开的情况下附加的情景
frida -U Twitter
frida -U 26984
frida -U -p 26984

# b. 启动应用进入交互模式, 应用于 App 未打开的情景
# 注意需要自己额外再输入 %resume，否则目标应用将一直处于暂停的状态
frida -U -f com.atebits.Tweetie2

# 如果启动应用后被强制退出或不想再额外输入 %resume，可以加上 --no-pause
frida -U -f com.atebits.Tweetie2 --no-pause
```

​        

## 2 基础API

```javascript
// 确保我们的线程附加到 Java 的虚拟机上，function 是成功之后的回调，之后的操作必须在这个回调里面进行，这也是 frida 的硬性要求
Java.perfom(function(){})

Java.use(className) //动态获取一个 JS 包装了的 Java 类
        $new()  // 通过$new方法来调用这个类构造方法
        $dispose()  // 最后可以通过$dispose()方法来清空这个 JS 对象

// 将 JS 包装类的实例 handle 转换成另一种 JS 包装类klass        
Java.cast(handle, kclass)

// 当获取到 Java 类之后，我们直接通过<wrapper>.<method>.implementations = function(){} 的方式来 hook wrapper 类的 method 方法，不管是实例方法还是静态方法都可以
var SQL = Java.use("com.xxx.xxx.database.SQLiteDatabase");
var ContentVaules = Java.use("android.content.ContentValues");
SQL.insert.implementation = function(arg1, arg2, arg3) {
    var values = Java.cast(arg3, ContentVaules);
}

// 由于 js 代码注入时可能会出现超时的错误，为了防止这个问题，我们通常还需要在最外面包装一层 setImmediate(function(){})代码
setImmediate(function(){
    Java.perform(function(){
        // start hook
        ...
    })
})
```



API 列表：

* `Java.choose(className: string, callbacks: Java.ChooseCallbacks): void` 通过扫描Java VM的堆来枚举className类的live instance；
* `Java.use(className: string): Java.Wrapper<{}>` 动态为className生成JavaScript Wrapper，可以通过调用`$new()`来调用构造函数来实例化对象。 在实例上调用`$dispose()`以对其进行显式清理，或者等待JavaScript对象被gc；
* `Java.perform(fn: () => void): void` Function to run while attached to the VM. Ensures that the current thread is attached to the VM and calls fn. (This isn't necessary in callbacks from Java.) Will defer calling fn if the app's class loader is not available yet. Use Java.performNow() if access to the app's classes is not needed；
* `send(message: any, data?: ArrayBuffer | number[]): void` 任何JSON可序列化的值。 将JSON序列化后的message发送到您的基于Frida的应用程序，并包含(可选)一些原始二进制数据。 The latter is useful if you e.g. dumped some memory using NativePointer#readByteArray()；
* `recv(callback: MessageCallback): MessageRecvOperation` Requests callback to be called on the next message received from your Frida-based application. This will only give you one message, so you need to call recv() again to receive the next one；
* `wait(): void` 堵塞，直到message已经receive并且callback已经执行完毕并返回

​       

# 附录



## 基础

1. [Github: frida框架详解](https://github.com/hookmaster/frida-all-in-one)
2. [苹果APP逆向神器Frida详细使用教程](https://www.vlwx.com/538.html)
3. [frida官网](https://frida.re/)
3. [Frida 常用 API 大全](https://www.jianshu.com/p/060bc0e91fdf)
3. [这恐怕是学习Frida最详细的笔记了](https://juejin.cn/post/6847902219757420552)
3. [Frida 二次开发工具 objection](https://book.hacktricks.xyz/mobile-pentesting/ios-pentesting/ios-hooking-with-objection)
3. [Frida 安装和使用](https://www.jianshu.com/p/bab4f4714d98)
3. [frida的用法--Hook Java代码篇](https://www.jianshu.com/p/f98aca8f3c05)

​     

## iOS

1. [iOS应用安全 —— 非越狱下使用Frida](https://mabin004.github.io/2018/06/18/iOS%E5%BA%94%E7%94%A8%E5%AE%89%E5%85%A8-%E2%80%94%E2%80%94-%E9%9D%9E%E8%B6%8A%E7%8B%B1%E4%B8%8B%E4%BD%BF%E7%94%A8Frida/)
2. [iOS逆向 - 运行时分析（三）Frida](https://juejin.cn/post/7079726534096846862)
3. [Frida官方手册 - 在iOS上使用Frida](https://zhuanlan.kanxue.com/article-345.htm)

​      

## Android

1. [Android Hook 之 Frida](https://mabin004.github.io/2018/03/09/droid-Hook-%E2%80%94%E2%80%94-Frida/)
2. [frida Android 的简单使用](https://juejin.cn/post/7008819110515736583)
3. [官网installation](https://frida.re/docs/android/)
4. [Frida 安装和使用: root和非root设备](https://www.jianshu.com/p/bab4f4714d98)
