Commands: 

Dowloading the images
````bash 
docker compose -f docker-compose_cluster.yml build
````

Up the docker containers
````bash
docker compose -f docker-compose_cluster.yml up -d
````


Things to check: 

 CMD ["bash"] in the Dockerfile does nothing useful. It's passed to start-spark.sh as an argument, and the script never reads arguments. This is harmless.


 start-env.sh is never used. It's only a template where everything is commented out. Nothing copies it, and Spark looks for a file called conf/spark-env.sh anyway

 I DONT KNOW OF THIS CHECK: BUT LEAVE?
 SPARK_MASTER_CORES: 2 (compose line 37) does nothing. Spark has no such setting and the script never reads it. You can remove it.

 Jupyter has no password set. 

 