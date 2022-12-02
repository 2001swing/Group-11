# 指定基础镜像
FROM python:3.7

#ENV server.params=

# updata太慢 设置镜像源
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && apt-get clean  \
    && apt-get update  \
    && apt-get install python3-dev default-libmysqlclient-dev -y


# 创建 code 文件夹并将其设置为工作目录
RUN mkdir /project
WORKDIR /project

# 将 requirements.txt 复制到容器
ADD requirements.txt /project/

# 更新 pip 并 安装依赖库
RUN pip install pip -U && pip install -r requirements.txt

# COPY 是不解压的
ADD . /project/

EXPOSE 8000

# CMD [ "/bin/sh", "/code/start.local.bat" ]
CMD ["/bin/sh","/project/runserver.sh"]