import os
import glob
import json
import subprocess
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='check the completeness of the corpus')
    parser.add_argument('--superpath',
                        default='slidespeech',
                        type=str,
                        help='the path to store slidespeech')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    superpath = args.superpath
    ips = json.load(open("./ips/ips.json", 'r'))["obj"]
    i = 0
    for group in ['train', 'dev', 'test']:
        with open(f"data/{group}/wavid2channel.json", 'r') as f:
            wavid2channel = json.load(f)
        mp4list = glob.glob(f"{superpath}/{group}/*/*.mp4")
        mp4listset = set([mp4path.split("/")[-1].split(".")[0] for mp4path in mp4list])
        all_list = set(list(wavid2channel.keys()))
        if len(all_list - mp4listset) == 0:
            print("{} {}/{} downloaded, finished.".format(group, len(mp4listset), len(all_list)))
        else:
            print("{} {}/{} downloaded, remains {}, not finished.".format(group, len(mp4listset), len(all_list), len(all_list - mp4listset)))
            new_downloading_file = f"data/{group}/process_fix.sh"
            print("downloading shell script: {}".format(new_downloading_file))
            with open(new_downloading_file, 'w') as f:
                for wavid in (all_list - mp4listset):
                    category = wavid.split("_")[0]
                    output_dir = f"{superpath}/{group}/{category}/{wavid}.%(ext)s"
                    youtube_channel = wavid2channel[wavid]
                    # while True:
                    # proxy = ips[i % len(ips)]["ip"] + ":" + ips[i % len(ips)]["port"]
                    process_str = "yt-dlp -vU -f 22 -o '{}' https://www.youtube.com/watch?v={}".format(
                        output_dir, youtube_channel
                    )
                        # try:
                    # subprocess.run(process_str, shell=True, check=True)
                        #     break
                        # except subprocess.CalledProcessError as e:
                        #     print(e)
                        #     print("proxy {} failed, try next.".format(proxy))
                        #     i += 1
                    f.write(f"{process_str}\n")
                    f.write(f"sleep 2m\n")

