
from compare import kitti, kiss_on_kitti, mad_on_kitti, mad_on_mulran, kitti_gt, mulran_gt

KITTI_BASE_DIR = "" # Is required for plotting on KITTI
MULRAN_BASE_DIR = ""
ODYSSYE_BASE_DIR = ""

KITTI_sequences = ["00","01","02","03","04","05","06","07","08","09","10"]
MulRan_sequneces = ["DCC01","DCC02","DCC03","Kaist01","Kaist02","Kaist03","Riverside01","Riverside02","Riverside03","Sejong01","Sejong02","Sejong03"]
Odyssey_sequences = ["Beltway1","Beltway2","Beltway3","CountryRoad1","CountryRoad2","CountryRoad3","ForestRoad1","ForestRoad2","ForestRoad3","Highway1","Highway2","Highway3", \
                    "HighwayTunnel1","HighwayTunnel2","HighwayTunnel3","InnerCity1","InnerCity2","InnerCity3","Marketplace1","Marketplace2","Marketplace3", \
                    "ParkingGarage1","ParkingGarage2","ParkingGarage3","Suburb1","Suburb2","Suburb3","Theater1","Theater2","Theater3","Tunnel1","Tunnel2","Tunnel3", \
                    "UndergroundCarPark1","UndergroundCarPark2","UndergroundCarPark3"]

def deduce_dataset_from_sequnece(sequence):
    if sequence in KITTI_sequences:
        return "kitti"
    elif sequence in MulRan_sequneces:
        return "mulran"
    elif sequence in Odyssey_sequences:
        return "odyssey"
    else:
        raise ValueError("Unrecognized sequence: ", sequence)

def load_poses(method,dataset,sequence):
    if method == "ours":
        path = "{0}/estimates/{1}/{2}/{3}.txt".format(base_dir,method,dataset,sequence)
        _, poses = kitti(path)
    elif method in ["kiss-icp","genz-icp","kiss-slam"]:
        path = glob.glob("{0}/estimates/{1}/{2}/*/{3}_poses_kitti.txt".format(base_dir,method,dataset,sequence))[0]
        if dataset == "kitti":
            _, poses = kiss_on_kitti(path,KITTI_BASE_DIR)
        else:
            _, poses = kitti(path)
    elif method == "mad-icp":
        path = "{0}/estimates/{1}/{2}/{3}/estimate.txt".format(base_dir,method,dataset,sequence)
        if dataset == "kitti":
            _, poses = mad_on_kitti(path,KITTI_BASE_DIR)
        elif dataset == "mulran":
            _, poses = mad_on_mulran(path)
        else:
            _, poses = kitti(path)
    else:
        raise ValueError("Unrecognized method: ", method)

    return poses


def load_gt_poses(dataset,sequence):
    if dataset == "kitti":
        path = f"{KITTI_BASE_DIR}/sequences/{sequence}.txt"
        _, poses = kitti_gt(path,KITTI_BASE_DIR)
    elif dataset == "mulran":
        path = f"{MULRAN_BASE_DIR}/{sequence}/global_pose.csv"
        _, poses = mulran_gt(path)
    else:
        path = f"{ODYSSYE_BASE_DIR}/{sequence}/refsys/lidar_poses_ref.txt"
        _, poses = kitti(path)

    return poses




if __name__ == "__main__":
    import glob
    import argparse
    import matplotlib.pyplot as plt
    from pathlib import Path
    base_dir = str(Path(__file__).parent.absolute())


    parser = argparse.ArgumentParser(
            prog='ProgramName',
            description='What the program does',
            epilog='Text at the bottom of help')

    parser.add_argument('sequence',type=str, default="00")
    parser.add_argument("-dataset",type=str,default="auto")

    args = parser.parse_args()
    sequence = args.sequence

    if args.dataset == "auto":
        dataset = deduce_dataset_from_sequnece(sequence)
    else:
        dataset = args.dataset

    try:
        gt_poses = load_gt_poses(dataset,sequence)
    except Exception as e:
        gt_poses = None
        pass

    ours_poses = load_poses("ours",dataset,sequence)
    kiss_icp_poses = load_poses("kiss-icp",dataset,sequence)
    mad_icp_poses = load_poses("mad-icp",dataset,sequence)
    genz_icp_poses = load_poses("genz-icp",dataset,sequence)
    kiss_slam_poses = load_poses("kiss-slam",dataset,sequence)

    if gt_poses is not None:
        plt.plot(gt_poses[:,0,-1],gt_poses[:,1,-1],label="GT",c="C0")

    plt.plot(ours_poses[:,0,-1],ours_poses[:,1,-1],label="Ours",c="C1")
    plt.plot(kiss_icp_poses[:,0,-1],kiss_icp_poses[:,1,-1],label="KISS-ICP",c="C2")
    plt.plot(mad_icp_poses[:,0,-1],mad_icp_poses[:,1,-1],label="MAD-ICP",c="C3")
    plt.plot(genz_icp_poses[:,0,-1],genz_icp_poses[:,1,-1],label="GenZ-ICP",c="C4")
    plt.plot(kiss_slam_poses[:,0,-1],kiss_slam_poses[:,1,-1],label="KISS-SLAM",c="C5")
    plt.legend()
    plt.show()


