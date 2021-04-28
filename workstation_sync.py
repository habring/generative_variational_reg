import sys
import os
#Python script to send or recieve files in a given folder or file to/from workstation
#The path of the folder/file is local in the /home/hollerm/ directory on both servers
# Use file or folder/ as input
#Example usage: python workstation_sync testfolder/
if __name__ == "__main__":

    #Get fname from argument
    fname = sys.argv[1]
    
    #IP and user
    #ip='143.50.47.123'
    #new
    ip='143.50.47.92'

    user='habring'


    #Basic sync command
    cmd = 'rsync -ave "ssh" -z --numeric-ids '
    
    remote = user + '@' + ip + ':/home/' + user + '/' + fname
    local = './' + 'receive_dir'
    
    #Select mode
    mode = int(input("Select mode: Send files (0), receive files (1): "))

    if mode == 0:
    
        msg = " SEND "
        #Add Laptop to Workstation command
        cmd +=  local + ' ' + remote
        
    elif mode == 1:

        msg = " RECEIVE "
        #Add Laptop to Workstation command
        cmd +=  remote + ' ' + local

    else:
        sys.exit("Error: Unknown mode")

        
    #Remove double //
    cmd = cmd.replace('//','/')
        
    print('Execute the following to' + msg + 'files??')
    print(cmd)

    #Get Confirmation    
    input('Press enter')

    os.system(cmd)



