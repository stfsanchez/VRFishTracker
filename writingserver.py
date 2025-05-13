"""
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: © 2022 Renaud Bastien, Stephane Sanchez <stephane.sanchez@ut-capitole.fr>
Université Toulouse Capitole - Institut de Recherche en Informatique de Toulouse
Real Expression Artificial Life (REVA) Research Team
https://www.irit.fr/departement/calcul-intensif-simulation-optimisation/reva/
"""
# Echo server program
import socket
import sys

TIMEOUTVALUE = 10

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432 #65434   # Port to listen on (non-privileged ports are > 1023)
SERVER = (HOST, PORT)
server =  socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(SERVER)

tracked_data = []
world_data = []

tracked_filename = ""
world_filename = ""
tracked_file = None
world_file = None

cpt = 0

#server.listen()

def receive():
    global tracked_data
    global world_data
    global tracked_filename
    global world_filename
    global tracked_file
    global world_file

    global cpt
    
    #client ,address =server.accept()
    #print(f'Connected with {str(address)} !')

    server.settimeout(30000.0)

    while True:
        try:
            data, addr = server.recvfrom(1024)
            sys.stdout.flush() 

            msg = data.decode('utf-8')
            
            if msg != "":
                if msg == "close":
                    print("---> Closing writing files")
                    if world_file is not None :
                        print("-> Closing world data file!")
                        world_file.close()
                        world_file = None
                    sys.exit(0)
                else:
                    cpt += 1
                    
                    if cpt % 30 == 0:
                        print(msg)
                    
                    msg_data = msg.split(" ") 

                    #print(msg_data)
                    
                    world_data = ""

                    #number of fish
                    fish = int(msg_data[0])
                    #file name
                    world_filename = msg_data[1]
                    #frame
                    world_data += msg_data[2] + " "
                    #time
                    world_data += msg_data[3]
                    #fish data
                    for i in range(0, fish*5):
                        world_data += " " + msg_data[4 + i]
                    
                    if world_file is None:
                        print("---> Opening " + world_filename)
                        try:
                            world_file = open(world_filename, 'a')
                        except:
                            new_filename = world_filename.split("/")
                            print("---> Creating " + new_filename[3])
                            world_file = open("./Data/" + new_filename[3], 'w')
                    world_file.write(world_data)
                    world_file.flush()
        except:     
            print("-------> No data --> exiting server !")
            if tracked_file is not None or world_file is not None:
                print("---> Closing writing files")
                if world_file is not None :
                    print("-> Closing world data file!")
                    world_file.close()
                    world_file = None
            sys.exit(0)
            break    
    sys.exit(0)


if __name__ == '__main__':
    print("------- Writing server is running ... -------")
    receive()