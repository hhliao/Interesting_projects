本文件是为了演示个人使用langchain-ChatGLM搭建个人知识库使用

操作步骤：
1. 解压缩langchain-ChatGLM.zip文件；
2. 运行：
    cd langchain-ChatGLM/
    conda create -n langchain-chatglm python=3.8.16
    conda activate langchain-chatglm
    pip install -r requirements.txt
    python demo_enpei.py
    静待模型下载完毕即可根据提示进行演示

3. 模型选择与参数配置
   - 所有的模型和参数配置在目录 config/model_config.py 中；
   - embedding模型的配置在该文件的15-22行，如果要在本地加载，则需要将对应的路径，可参考 text2vec, 该文件第25行指定使用的embedding模型；
   - LLM模型的配置在该文件的35-186行，默认是从网上下载对应的模型文件，如要在本地加载，可参考66行chatglm2-6b模型，在local_model_path中指定模型的路径即可；
   - demo_enpei.py文件为命令行演示的主文件。
