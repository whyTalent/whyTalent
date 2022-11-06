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

 `Android` / `iOS` 设备环境配置，以及APP的frida-gadget持久化配置，详情见 [frida APP逆向配置](3 frida APP逆向配置.md)

​         

# 三 firda工具&API

​       

## 1 Frida 基础工具

Frida 支持一下基础工具：

### 1.1 frida-ls-devices

查看可用的设备列表

   

### 1.2 frida-ps

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

### 1.3 frida-kill

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

### 1.4 frida-trace

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

### 1.5 frida CLI

交互模式

```shell
# 提供了两种进入交互模式的方式
# a. 通过应用名或 PID 附加, 应用于 App 已打开的情况下附加的情景
frida -U Twitter
frida -U 26984
frida -U -p 26984

# b. 启动应用进入交互模式, 应用于 App 未打开的情景
# 注意需要自己额外再输入 %resume，否则目标应用将一直处于暂停的状态
frida -U -f com.atebits.Tweetie2

# 如果启动应用后被强制退出或不想再额外输入 %resume，可以加上 --no-pause
frida -U -f com.atebits.Tweetie2 --no-pause

# attach 前台应用, 并注入js脚本执行
frida -U -F -l xxx.js

# 支持参数:
# -l 指定加载一个Javascript脚本
# -U 指定对USB设备操作
# -F 绑定前台应用

# frida特殊指令:
# 1) %load / %unload：载⼊/卸载⽂件中的 js
# 2) %reload：修改外部 js 之后重新载⼊脚本，且重置之前的 hook
# 3) %resume: %resume：继续执⾏以 spawn 模式启动的进程
# 4) quit/exit 或 Ctrl + D: 退出结束脚本注入
```

​     

### 1.6 frida-compile

使用 `frida-compile` 支持实时把成 `TypeScript` 编译成 `JavaScript` 代码

**注**：Frida不支持直接注入TypeScipt

```shell
# frida-compile
usage: frida-compile [options] <module>

positional arguments:
  module                TypeScript/JavaScript module to compile

optional arguments:
  -h, --help            show this help message and exit
  -O FILE, --options-file FILE
                        text file containing additional command line options
  --version             show program's version number and exit
  -o OUTPUT, --output OUTPUT
                        write output to <file>
  -w, --watch           watch for changes and recompile
  -S, --no-source-maps  omit source-maps
  -c, --compress        compress using terser
  -v, --verbose         be verbose
```

```shell
# demo
frida-compile agent/android.ts -o _android.js -c
frida-compile agent/android.ts -o _android.js -w
```

​         

## 2 Frida API

主要包括 Java、Interceptor对象、NativePointer对象等

​      

### 2.1 Java对象API 

下面简单介绍场景的Java 对象

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

​    

#### 1）Java.available & Java.androidVersion

`available`：判断 Java 环境可⽤，即这个字段标记 Java虚拟机（例如： Dalvik 或者 ART）是否已加载， 操作Java任何东西之前，要确认这个值是否为true

`androidVersion`：显示android系统版本号

```typescript
function frida_Java() {
    Java.perform(function () {
        //作为判断用
        if(Java.available) {
            // 注入的逻辑代码
            console.log("hello java vm");
            console.log("", Java.androidVersion);
        } else {
            // 未能正常加载JAVA VM
            console.log("error");
        }
    });
}       
setImmediate(frida_Java,0);
// setTimeout(frida_Java,1000);
```

​    

#### 2）枚举加载类 Java.enumerateLoadedClasses(callbacks)

枚举当前加载的所有类class信息，它有一个回调函数分别是 `onMatch`、`onComplete`函数

`Java.enumerateLoadedClassesSync()`：同步返回类class信息数组

```typescript
Java.perform(function () {
    if(Java.available){
        //枚举当前加载的所有类
        Java.enumerateLoadedClasses({
            //每一次回调此函数时其参数className就是类的信息
            onMatch: function (className) {
                //输出类字符串
                console.log("", className);
            },
            //枚举完毕所有类之后的回调函数
            onComplete: function () {
                //输出类字符串
                console.log("输出完毕");
            }
        });
    } else {
        console.log("error");
    }
});
```

​      

#### 3）枚举类加载器 Java.enumerateClassLoaders

枚举`Java VM`中存在的类加载器，其有一个回调函数，分别是`onMatch: function (loader)`与`onComplete: function ()`

`Java.enumerateClassLoadersSync()`：返回类加载器数组

```typescript
Java.perform(function () {
    if(Java.available) {
        //枚举当前加载的Java VM类加载器
        Java.enumerateClassLoaders({
            //回调函数，参数loader是类加载的信息
            onMatch: function (loader) {
                console.log("",loader);
            },
            //枚举完毕所有类加载器之后的回调函数
            onComplete: function () {
                console.log("end");
            }
        });
    }else{
        console.log("error");
    }
});
```

​      

#### 4）附加调用 Java.perform

`Java.perform（fn）`主要用于当前线程附加到`Java VM`，并且调用`fn`方法，注意执⾏任意 Java 操作都需要使⽤此函数

>1）ART 和 Dalvik 都按照 JVM 的规范实现
>
>2）frida 的 js 脚本引擎使⽤了（⾮主线程）的其他线程，需要使⽤ javaVM->AttachCurrentThread，⽽对应为了释放资源，完成任务后 需 DetachCurrentThread
>
>3）为了保证关联和释放，所有涉及 JVM 的操作都需要放在 Java.perform 回调中执⾏

```typescript
setTimeout(function() {
    // 运行当前js脚本时会对当前线程附加到Java VM虚拟机，并且执行function方法
    Java.perform(function () {
        //判断是否Java VM正常运行
        if(Java.available) {
            //如不意外会直接输出 hello
            console.log("hello");
        } else {
            console.log("error");
        }
        
        const Activity = Java.use('android.app.Activity');
        Activity.onResume.implementation = function () {
            send('onResume() got called! Let\'s call the original implementation');
            this.onResume();
        };
    });
}, 1000);
```

​         

#### 5）获取类 Java.use

`Java.use(className)`：动态获取 `className` 的类定义，通过对其调用`$new()`来调用构造函数，可以从中实例化对象。当想要回收类时可以调用`$Dispose()`方法显式释放，当然也可以等待`JavaScript`的垃圾回收机制，当实例化一个对象之后，可以通过其实例对象调用类中的静态或非静态的方法。

```typescript
Java.perform(function () {
  //获取android.app.Activity类
  var Activity = Java.use('android.app.Activity');
  //获取java.lang.Exception类
  var Exception = Java.use('java.lang.Exception');
  // 加载内部类
  var MyClass_InnerClass = Java.use("com.luoyesiqiu.MyClass$InnerClass");
  //拦截Activity类的onResume方法
  Activity.onResume.implementation = function () {
    //调用onResume方法的时候，会在此处被拦截并且调用以下代码抛出异常！
    throw Exception.$new('Oh noes!');
  };
});
```

​      

#### 6）扫描实例类 Java.choose

在内存-堆上查找实例化的对象

```typescript
Java.perform(function () {
    //查找android.view.View类在堆上的实例化对象
    Java.choose("android.view.View", {
        //枚举时调用
        onMatch:function(instance){
            //打印实例
            console.log("[*] onMatch: " + instance);
        },
        //枚举完成后调用
        onComplete:function() {
            console.log("end")
        }});
});
```

​       

#### 7）类型转换器 Java.cast

`Java.cast(handle, klass)`：就是将指定变量或者数据强制转换成你所有需要的类型；创建一个 `JavaScript` 包装器，给定从 `Java.use（）` 返回的给定类`klas`的句柄的现有实例。

此类包装器还具有用于获取其类的包装器的类属性，以及用于获取其类名的字符串表示的`$className`属性，通常在拦截`so`层时会使用此函数将`jstring、jarray`等等转换之后查看其值。

```typescript
// 将 variable 转换成java.lang.String
var StringClass=Java.use("java.lang.String");
var NewTypeClass=Java.cast(variable,StringClass);

const Activity = Java.use('android.app.Activity');
const activity = Java.cast(ptr('0x1234'), Activity);
```

​      

#### 8）主线程执行 

`scheduleOnMainThread`：在 JVM 主线程执⾏⼀段函数       

​    

#### 9）Java.vm

Java.vm对象十分常用，比如想要拿到JNI层的JNIEnv对象，可以使用 getEnv()

```typescript
function frida_Java() {     
    Java.perform(function () {
         //拦截getStr函数
         Interceptor.attach(Module.findExportByName("libhello.so" , "Java_com_roysue_roysueapplication_hellojni_getStr"), {
            onEnter: function(args) {
                console.log("getStr");
            },
            onLeave:function(retval){
                //它的返回值的是retval 在jni层getStr的返回值的jstring 
                //我们在这里做的事情就是替换掉结果
                //先获取一个Env对象
                var env = Java.vm.getEnv();
                //通过newStringUtf方法构建一个jstirng字符串
                var jstring = env.newStringUtf('roysue');
                //replace替换掉结果
                retval.replace(jstring);
                console.log("getSum方法返回值为:roysue")
            }
    });
}
   
setImmediate(frida_Java,0);
```

​      

#### 10）注册类Java.registerClass(spec)

`Java.registerClass`：创建一个新的`Java`类并返回一个包装器，其中规范是一个包含：

> `name`：指定类名称的字符串；
>
> `superClass`：（可选）父类。要从 `java.lang.Objec`t 继承的省略；
>
> `implements`：（可选）由此类实现的接口数组；
>
> `fields`：（可选）对象，指定要公开的每个字段的名称和类型；
>
> `methods`：（可选）对象，指定要实现的方法；

```typescript
//获取目标进程的SomeBaseClass类
var SomeBaseClass = Java.use('com.example.SomeBaseClass');
//获取目标进程的X509TrustManager类
var X509TrustManager = Java.use('javax.net.ssl.X509TrustManager');

var MyWeirdTrustManager = Java.registerClass({
  //注册一个类是进程中的MyWeirdTrustManager类
  name: 'com.example.MyWeirdTrustManager',
  //父类是SomeBaseClass类
  superClass: SomeBaseClass,
  //实现了MyWeirdTrustManager接口类
  implements: [X509TrustManager],
  //类中的属性
  fields: {
    description: 'java.lang.String',
    limit: 'int',
  },
  //定义的方法
  methods: {
    //类的构造函数
    $init: function () {
      console.log('Constructor called');
    },
    //X509TrustManager接口中方法之一，该方法作用是检查客户端的证书
    checkClientTrusted: function (chain, authType) {
      console.log('checkClientTrusted');
    },
    //该方法检查服务器的证书，不信任时。在这里通过自己实现该方法，可以使之信任我们指定的任何证书。在实现该方法时，也可以简单的不做任何处理，即一个空的函数体，由于不会抛出异常，它就会信任任何证书。
    checkServerTrusted: [{
      //返回值类型
      returnType: 'void',
      //参数列表
      argumentTypes: ['[Ljava.security.cert.X509Certificate;', 'java.lang.String'],
      //实现方法
      implementation: function (chain, authType) {
         //输出
        console.log('checkServerTrusted A');
      }
    }],
    //　返回受信任的X509证书数组。
    getAcceptedIssuers: function () {
      console.log('getAcceptedIssuers');
      return [];
    },
  }
});
```

​        

### 2.2 Interceptor对象

该对象功能十分强大，函数原型是 `Interceptor.attach(target, callbacks)`

> `target` 参数是需要拦截的位置的函数地址，也就是填某个`so`层函数的地址即可对其拦截，`target`是一个`NativePointer`参数，用来指定你想要拦截的函数的地址，`NativePointer`是一个指针。需要注意的是对于`Thumb`函数需要对函数地址`+1`；
>
> `callbacks`则是它的回调函数，分别是以下两个回调函数：
>
> * `onEnter：`函数（`args`）：回调函数，给定一个参数`args`，可用于读取或写入参数作为 `NativePointer` 对象的数组
>
> * `onLeave：`函数（`retval`）：回调函数给定一个参数 `retval`，该参数是包含原始返回值的 `NativePointer` 派生对象。可以调用 `retval.replace（1337）` 以整数 `1337` 替换返回值，或者调用 `retval.replace（ptr（"0x1234"））`以替换为指针
>
>   请注意，此对象在 `OnLeave` 调用中回收，因此不要将其存储在回调之外并使用它。如果需要存储包含的值，请制作深副本，例如：`ptr（retval.toString（））`。

​      

#### 1）Interceptor.attach

函数属性：

> returnAddress：返回地址，类型是NativePointer
>
> context：上下文，即具有键pc和sp的对象，它们是分别为ia32/x64/arm指定EIP/RIP/PC和ESP/RSP/SP的NativePointer对象。其他处理器特定的键也可用，例如eax、rax、r0、x0等。也可以通过分配给这些键来更新寄存器值
>
> errno：当前errno值
>
> lastError：当前操作系统错误值
>
> threadId：操作系统线程ID
>
> depth：相对于其他调用的调用深度

```typescript
// 对So层的导出函数getSum进行拦截
//对So层的导出函数getSum进行拦截
Interceptor.attach(Module.findExportByName("libhello.so" , "Java_com_roysue_roysueapplication_hellojni_getSum"), {
    onEnter: function(args) {
        //输出
        console.log('Context information:');
        //输出上下文因其是一个Objection对象，需要它进行接送、转换才能正常看到值
        console.log('Context  : ' + JSON.stringify(this.context));
        //输出返回地址
        console.log('Return   : ' + this.returnAddress);
        //输出线程id
        console.log('ThreadId : ' + this.threadId);
        console.log('Depth    : ' + this.depth);
        console.log('Errornr  : ' + this.err);
    },
    onLeave:function(retval){
        if (retval.toInt32() > 0) {
            /* do something with this.fileDescriptor */
        }
    }
});

// 使用Module对象getExportByNameAPI直接获取libc.so中的导出函数read的地址，对read函数进行附加拦截
Interceptor.attach(Module.getExportByName('libc.so', 'read'), {
    ... ...
});
```

​     

#### 2）Interceptor.detachAll

函数的作用就是让之前所有的 `Interceptor.attach` 附加拦截的回调函数失效

​    

#### 3）Interceptor.replace

相当于替换掉原本的函数，用替换时的实现替换目标处的函数。如果想要完全或部分替换现有函数的实现，则通常使用此函数

```typescript
Java.perform(function () {
    //这个c_getSum方法有两个int参数、返回结果为两个参数相加
    //这里用NativeFunction函数自己定义了一个c_getSum函数
    var add_method = new NativeFunction(Module.findExportByName('libhello.so', 'c_getSum'), 'int',['int','int']);
    
    //输出结果 那结果肯定就是 3
    console.log("result:",add_method(1,2));
    //这里对原函数的功能进行替换实现
    Interceptor.replace(add_method, new NativeCallback(function (a, b) {
        //h不论是什么参数都返回123
        return 123;
    }, 'int', ['int', 'int']));
    
    //再次调用 则返回123
    console.log("result:",add_method(1,2));
});
```

​       

### 2.3 NativePointer对象

同等与C语言中的指针

```typescript
function frida_NativePointer() {
    Java.perform(function () {
        // 第一种字符串定义方式 十进制的100 输出为十六进制0x64
        const ptr1 = new NativePointer("100");
        console.log("ptr1:",ptr1);
        
        // 第二种字符串定义方式 直接定义0x64 同等与定义十六进制的64
        const ptr2 = new NativePointer("0x64");
        console.log("ptr2:",ptr2);      
        
        // 第三种定数值义方式 定义数字int类型 十进制的100 是0x64
        const ptr3 = new NativePointer(100);
        console.log("ptr3:",ptr3);
    });
}     

setImmediate(frida_NativePointer,0);
```

​       

### 2.4 NativeFunction对象

创建新的`NativeFunction`以调用`address`处的函数（用`NativePointer`指定），其中`rereturn Type`指定返回类型，`argTypes`数组指定参数类型。如果不是系统默认值，还可以选择指定`ABI`。对于可变函数，添加一个‘.’固定参数和可变参数之间的`argTypes`条目

```typescript
// LargeObject HandyClass::friendlyFunctionName();

//创建friendlyFunctionPtr地址的函数
var friendlyFunctionName = new NativeFunction(friendlyFunctionPtr,
    'void', ['pointer', 'pointer']);

//申请内存空间    
var returnValue = Memory.alloc(sizeOfLargeObject);

//调用friendlyFunctionName函数
friendlyFunctionName(returnValue, thisPtr);
```

函数定义格式为`new NativeFunction(address, returnType, argTypes[, options])，`参照这个格式能够创建函数并且调用`！returnType和argTypes[，]`分别可以填`void、pointer、int、uint、long、ulong、char、uchar、float、double、int8、uint8、int16、uint16、int32、uint32、int64、uint64`这些类型，根据函数的所需要的type来定义即可。

在定义的时候必须要将参数类型个数和参数类型以及返回值完全匹配，假设有三个参数都是`int`，则`new NativeFunction(address, returnType, ['int', 'int', 'int'])`，而返回值是`int`则`new NativeFunction(address, 'int', argTypes[, options])`，必须要全部匹配，并且第一个参数一定要是函数地址指针。

​         

### 2.5 NativeCallback对象

`new NativeCallback(func，rereturn Type，argTypes[，ABI])：`创建一个由`JavaScript`函数`func`实现的新`NativeCallback`，其中`rereturn Type`指定返回类型，`argTypes`数组指定参数类型。您还可以指定`ABI`(如果不是系统默认值)。

注意，返回的对象也是一个`NativePointer`，因此可以传递给`Interceptor#replace`。当将产生的回调与`Interceptor.replace()`一起使用时，将调用func，并将其绑定到具有一些有用属性的对象，就像`Interceptor.Attach()`中的那样。

```typescript
Java.perform(function () {
    var add_method = new NativeFunction(Module.findExportByName('libhello.so', 'c_getSum'), 'int',['int','int']);
    console.log("result:",add_method(1,2));
    
    //在这里new一个新的函数，但是参数的个数和返回值必须对应
    Interceptor.replace(add_method, new NativeCallback(function (a, b) {
        return 123;
    }, 'int', ['int', 'int']));
    console.log("result:",add_method(1,2));
});
```

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
3. [FRIDA-API：Java、Interceptor、NativePointer(Function/Callback)](https://www.anquanke.com/post/id/195869#h2-13)
3. [Frida工作原理学习(1)](https://bbs.pediy.com/thread-273450.htm)

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
4. [app逆向基础 安装frida框架](https://codeantenna.com/a/0EQwO1RM6N)

