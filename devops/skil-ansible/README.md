Create a host list in the ".ssh/config" to configure the hosts before executing the scripts.

A sample host file will look like this: 
```text
Host server1
 HostName 10.0.0.100
 User myuser
 Port 22
 IdentityFile ~/.ssh/id_rsa
 
 Host server2
  HostName 10.0.0.101
  User myuser
  Port 22
  IdentityFile ~/.ssh/id_rsa
```

