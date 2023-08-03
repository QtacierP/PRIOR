from multiprocessing.connection import wait
import subprocess
from config import configs
import logging


log = logging.getLogger(__name__)

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map



if __name__ == '__main__':
    threshold = configs['gpu_threshold']
    job_list = configs['job_list']
    gpu_min_num = configs['gpu_min_num']
    current_job_ldx = 0
    wait_print = True
    while True:
        gpu_stats = get_gpu_memory_map()
        if wait_print:
            print('[INFO] waiting for gpu usage to run {} ...'.format(job_list[current_job_ldx]))
            wait_print = False
        gpu_list = []
        for k, v in gpu_stats .items():
            if v <= threshold:
                gpu_list.append(k)
        if len(gpu_list) >= gpu_min_num:
            print('[INFO] find gpu:', gpu_list)
        else:
            continue
        job = job_list[current_job_ldx] + ' +trainer.gpus=' + str(gpu_list).replace(' ', '')
        print('run job ', job)
        child = subprocess.Popen(job, shell=True)
        child.wait()
        current_job_ldx += 1
        wait_print = True
        if current_job_ldx >= len(job_list):
            print('[INFO] all jobs done')
            break

        

    
    
        

