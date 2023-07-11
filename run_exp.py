
import sys
from behav_runner.experiment.Experiment import Experiment
import params

class Subject:

    def __init__(self, id, age, gender, hand):
        self.__id = id
        self.__age = age
        self.__gender = gender
        self.__hand = hand

    def get_id(self):
        return self.__id

    def get_age(self):
        return self.__age

    def get_gender(self):
        return self.__gender

    def get_hand(self):
        return self.__hand

def main():


    print("hello world")
    try:
        subject_id = sys.argv[1]
        age = sys.argv[2]
        gender  = sys.argv[3]
        eye = sys.argv[4]
    except:
        print ("please enter subject id , age, gender, dominant hand")
        exit()

    print ("subject id is %s " %(subject_id))

    current_subject = Subject(subject_id, age, gender, eye)
    dummy_mode = True
    run_calib = False
    current_experiment = Experiment(current_subject, dummy_mode=dummy_mode, run_calib = run_calib)

    # if current_experiment.is_id_exists(subject_id):
    #     print("SUBJECT EXISTS!!")
    #     ok = yesNo("Subject id already exists. is it ok? continue? (y/n):")
    #     if not ok:
    #         print ("USE NEW ID")
    #         exit()

    tasks = current_experiment.run_depression_behav_exp(word_probe_task=True, sen_task = True)


    [current_experiment.export_task_data(tsk) for tsk in tasks]


    print("bye world")
    current_experiment.close_et_final()
    print("Exiting")
    exit()

def yesNo(q):
    inp = input(q)
    reply = str(inp).lower().strip()
    if reply[0] == 'y':
        print("OK - continue")
        return True
    if reply[0] == 'n':
        return False
    else:
        return yesNo("Uhhhh... please enter ")

if __name__ == '__main__':
    main()