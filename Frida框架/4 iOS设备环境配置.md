# iOS-Frida 环境配置

下面简单记录frida在 iOS 上使用 `frida-server` 和 `frida-gadget` 2种环境配置方式，为后续进一步学习实践~

​       

# 一 基础知识

在iOS设备上，Frida支持两种使用模式，具体使用哪种模式要看你的iOS设备是否已经越狱

​       

### 2.1 已越狱设备

越狱机上使用Cydia工具配置Frida

> 1）启动 Cydia
>
> 2）添加软件源：manage -> 软件源 Sources-> 编辑 Edit（左上角）-> 添加 Add（右上角）-> 输入 https://build.frida.re/
>
> 3）通过刚才添加的软件源安装 frida 插件，注意需要根据手机进行安装：iPhone 5 及之前的机器为 32 位，5s 及之后的机器为 64 位，进入 变更 -> 找到Frida -> 进入Frida 在右上角点击安装

​       

### 2.2 未越狱设备

frida-server在运行时需要root环境，但如果没有越狱的设备，依然可以使用frida，只需要重打包ipa文件，将frida运行库注入ipa文件中，app在启动时会自动加载frida运行库，即可实现在非越狱的设备上使用Frida。

因此，为了让一个App能使用Frida，必须想办法让它加载一个 **.dylib**，就是一个 **Gadget** 模块，因此需要配置一下 **xcode** 的编译配置来让你的App可以集成Frida。当然也可以使用相关的工具来修改一个已经编译好的App， 比如 **insert_dylib** 这样的工具。

... ...

​      

### 2.3 模拟器

在模拟器中进行测试，需要把命令行中的 **-U** 替换成 **-R**，这样一来底层的内部调用也从 **get_usb_device()** 变成 **get_remote_device()**

... ...