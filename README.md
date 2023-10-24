# dig-dig-books
数据挖掘 - 基于书籍文本属性与链接关系的类别预测信息收集

## 1.环境

和==之前一样==的,用之前的机器学习弄就好(另外俩人面基我帮弄)

还是conda,还是python3.9.16版本

目前已经装了的包有

```
torch
torchvision
jupyter
jinjia
matplotlib
d2l (这个是李沐那个,有没有都无所谓)
```

**后面有新加的包补充到这里就可以**

### 1.1 新加的包

```
# 新要安装的包这里说明就可以
比如说 pip install pillow
```

## 2.gitignore

因为数据集太大,不塞数据集上去,在你的文件夹下面创建一个dataset,然后把数据集塞进去(我发群里的直接解压到这里面就可,`.gitignore`会自动忽略),后续生成的数据集也保存在dataset里面

如果有不懂的群里问我就好

## 3. 合作

方便起见,就不创建分支了(怕大家不好合并)

**第一次打开时,使用`git clone "url"`**,然后你当前目录下就会多个`dig-dig-books` 文件夹

`url `在这个位置

![image-20231024232849772](https://zzhaire-markdown.oss-cn-shanghai.aliyuncs.com/img/image-20231024232849772.png)

每次打开文件夹前,你需要做的

1.  `git pull` 拉取远程仓库最新代码(一定要先拉,没拉就只能删除文件夹重新`git clone`了,解决冲突很麻烦)
2.  然后开始你的表演
3.  完成你的表演
4.  `git add .` 添加所有的更改到TEMP
5.  `git commit -m"这次更改你所要告诉大家的信息"` 把更改保存到本地git管理
6.  `git push` 把你的更改推送到远程仓库

~~如果大家会用git当我没说~~

**然后我会把大家的github账户拉进来,就可以一起写代码了**

祝大家挖掘顺利,~~挖不粗来就女装~~

## 4. 书写规范

## 4.1 开发说明

最后把写好的东西放到code里面,要提交的最终的核心代码都放在 main.ipynb,大家可以在文件夹`code/` 里面随便创建东西,外面就不要放了,最好自己的琐碎工作可以单独创建文件夹.

## 4.2 数据集规范说明

>   保存的中间变量(凡是比较大的东西,比如说洗好的数据,或者保存的模型)都放到`dataset/`文件夹下
>
>   github不会上传这些文件,不然会导致每次拉代码都非常慢 
>
>   **生成必要文件的代码放到main.ipynb(当前文件),注释写好,避免大家到时候还要在群里传东西** 

**示例1**

``` py
# ! 必须要执行的代码
pd.save("../dataset/clean_data.csv") #这个是xxx洗好的数据,可以用pd.read("../dataset/clean_data.csv")读取它
```
## 4.3 注释标准说明

不需要说太多自己干了什么,只需要在对接的时候,告诉别人怎么用,以及可能出现的问题.(自己想干啥可以在外面创建`ipynb`或者`py`文件自己搞,main里只放可以生成必要东西的精简代码)
- 必须要执行代码注释用! 标记 例如 `# ! 这是必须要执行的代码`
- 有歧义的代码用 ? 标记 例如 `# ? 我这里为什么会报错,是我中间哪里没执行吗`
- 未完成的工作用 TODO 标记 例如 `# TODO : 这里暂时还没写完`

**示例1**

``` python
# TODO: 数据标准化函数还没弄完
def standardizing( data ):
    ...
    return stand_data
```
**示例2**

``` python
# !必须要执行的指令,如果已经于10月24日后保存此文件,则请跳过本cell
data_stand = standardizing(data)
pd.save("../dataset/clean_data.csv") #标准化后的数据,读取时注意数据下溢的问题
# !必须
my_data = pd.read("../dataset/clean_data.csv")
```
**示例3**

```python
# ? 编码问题,读出来是乱码
data = pd.read("../dataset/clean_data.csv")
data
```

-   
    -   
