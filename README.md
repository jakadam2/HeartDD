Starting the app:

Application requires Tkinter to be installed on your PC. It will not be installed automatically as it is not a pip library

If you wish to start the client and the server on the same machine you can do that by running:
```
python start.py
```
Otherwise use
```
python client/client.py    - to start the client
python server/server.py    - to start the server
```

To make testing easier you can change the "testing" values in client.test and server.test in config.ini file to "yes"

line at the end of Client contructor in Client.py. This will automatically load a test file.