from Features import feat
from buzzertutorial import buzz
from Id3 import train

def main():
    while True:    
        if(train()):
            buzz()
main()
    
    