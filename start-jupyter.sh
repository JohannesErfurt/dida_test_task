/usr/bin/python3.10 /usr/local/bin/jupyter-notebook --ip 0.0.0.0 --port 8888 --notebook-dir=notebooks --no-browser --allow-root &
sleep 3
jupyter notebook list > jupyter-token.txt

