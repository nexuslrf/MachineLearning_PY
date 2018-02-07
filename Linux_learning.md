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

