1. 安装jupyter notebook

   首先，安装anaconda软件包，然后激活相应的python环境后，运行以下命令：
   conda upgrade --all  # 更新所有软件包
   conda install jupyter notebook # 安装notebook相关


2. 运行已编写好的notebook

   在上述步骤安装好jupyter notebook后，在命令行执行：
   cd /path to fun-tuning_chatglm2-6b/
   unzip chinese_food_dataset.zip 
   jupyter notebook
   会自动打开浏览器，在浏览器上可以看到enpei_class目录下的所有文件，点击 ChatGLM2-6B调优中文菜谱.ipynb 即可运行notebook.


  注意：
  1. 在使用notebook时一定要注意notebook的内核一定是 conda 激活的环境；
  2. 在训练过程中，GPU现存消耗大概在15G左右，如果现存不够，可以适当调小batch_size的大小（在cell 22处），目前是4；
  3. peft工具包的介绍见我的个人主页。


演示最后结果：

1. 解压缩meishi_chatglm2_qlora.zip；
2. 打开ChatGLM2-6B调优中文菜谱.ipynb，直接进入到【5. 验证模型】部分，运行即可得到最终结果