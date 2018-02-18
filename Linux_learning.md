# Linux learning

## Basic operation

### cd
Change Directory Terminal 中的 ~ $ 就是说你输入指令将在 ~ 这个目录下执行. 而 ~ 这个符号代表的是你的 Home 目录. 如果在文件管理器中可视化出来, 就是下面图中那样.
![img](https://morvanzhou.github.io/static/results/linux-basic/02-01-02.png)
* 其他常用command
  1. 返回上一级
  >cd ..
  2. 返回你刚刚所在的目录
  >cd -
  3. 向上返回两次
  >cd ../../
  4. 去往 Home
  >cd ~

### ls (list)
1. ls -l  输出详细信息(long)
2. ls -a 显示所有文件(all)
3. ls -lh 方便人观看(human)
  ... ls --help

### touch(新建)
>touch file1 file2 file3

### cp(复制)
>cp old new
>cp old folder/

-i (interactive) 注意: 如果 file1copy 已经存在, 它将会直接覆盖已存在的 file1copy, 如果要避免直接覆盖, 我们在 cp 后面加一个选项.
>cp -i file1 file1copy

复制去文件夹
>cp file folder/

复制文件夹，需加上 -R(recursive)
>cp -R folder1/ folder2/

### mv(剪切)

1. 移去另一个文件夹
>mv file folder/
2. 重命名文件夹
>mv file1 file1rename

### mkdir（make directory）

> mkdir folder1

> mkdir folder/f1

### rmdir (remove directory)

移除文件夹. 不过这有一个前提条件. 这些要移除的文件夹必须是空的. 不然会失败. 

> rmdir folder

### rm (remove file)

> rm file

`-i` interactive mode

> rm -i f1 f2 f3

`-r` (recursively) 删除文件夹

> rm -r folder

### nano (a file editor)

> touch t.py
>
> nano t.py

### cat (catenate) 

可以用来显示文件内容, 或者是将某个文件里的内容写入到其他文件里

> cat t.py

`>` 将文件的内容放到另一个文件里

> cat t.py>t1.py

> cat t.py t1.py>t2.py

`>>` 将内容添加在一个文件末尾

> cat t.py>>t2.py

## 文件权限

### 查看权限

> ls -l

![img](https://morvanzhou.github.io/static/results/linux-basic/03-01-02.png)

- `Type`: 很多种 (最常见的是 `-` 为文件, `d` 为文件夹, 其他的还有`l`, `n` … 这种东西, 真正自己遇到了, 网上再搜就好, 一次性说太多记不住的).
- `User`: 后面跟着的三个空是使用 User 的身份能对这个做什么处理 (`r` 能读; `w` 能写; `x` 能执行; `-` 不能完成某个操作).
- `Group`: 一个 Group 里可能有一个或多个 user, 这些权限的样式和 User 一样.
- `Others`: 除了 User 和 Group 以外人的权限.

### chmod 修改权限 

> chmod \[who (ugo)] \[how to change] \[which file]

> chmod u+r t1.py

> chmod ug+x t1.py

> chmod o-w t1.py

## 使用 Python 的技巧 

我不怎么用权限这东西, 但是我却发现给 python 文件添加权限 `x` 还算有用的. 为什么这么说? 因为通常, 如果一个 `.py` 没有 `x` 权限, 在 terminal 中你就需要这样执行:

```bash
$ python3 t.py
This is a Python script!

```

如果你有了 `x` (可执行权限), 你运行这个文件可以直接这样打:

```bash
$ ./t.py
This is a Python script!

```

如果你天天要运行这个脚本, 每次运行的时候少几个字还是挺好的. 如果你决定要这样做, 你在这个 Python 脚本的开头还需要加一句话.

```bash
#!/usr/bin/python3        # 这句话是为了告诉你的电脑执行这个文件的时候用什么来加载

print("This is a Python script!")
```

## C++ 方法(intro)

> touch hello.cpp

> vi hello.cpp

随便写一个hello world~

#### g++编译

> g++ f1.cpp

生成默认为a.out的文件，这个过程包括编译和链接

> g++ f1.cpp -o ans.out

`-o` 是输出(out) 的意思