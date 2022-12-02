docker stop photo1
docker rm photo1
docker run -dit -p 8001:8000 --name photo1 photodenoising
