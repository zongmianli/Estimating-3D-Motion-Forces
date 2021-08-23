import numpy as np
import cPickle as pk


def load_raw_contact_states(pkl_path, video_name, joint_mapping,
                            frame_start, frame_end):
    '''
    Loads raw contact states as a 24 x nf array whose columns are the contact
    states of the 24 joints considered in our human model.
    Note that the we recognize the contact states of only 9 joints (neck,
    left/right hands, knees, soles and toes) among the 24 joints.
    Therefore, in the output contact_states array, only the rows corresponding
    to these 9 joints contain information, while the others are all zero.
    '''

    with open(pkl_path, 'r') as f:
        data = pk.load(f)
        item_id = data['item_names'].index(video_name)
        preds = data['contact_states'][item_id][(frame_start-1):frame_end]

    # Convert preds (nf x 9 x 4 array) to contact_states (24 x nf array)
    nf = preds.shape[0]
    contact_states = np.zeros((24, nf)).astype(int)

    for j, jid in enumerate(joint_mapping):
        # j: joint id in contact states
        # jid: joint id in our human model
        for fid in range(nf):
            pred = preds[fid, j, :].tolist().index(1)
            # Possible pred values:
            # 0 - joint in contact
            # 1 - joint not in contact
            # 2 - joint occluded
            # 3 - joint undetected by Openpose
            if pred==0:
                # We assume that joints of upper limb (neck, hands) can only
                # have object contact, and joints of lower limb (knees, soles,
                # toes) can only be in contact with the ground.
                if jid in [12, 18, 23]:
                    contact_states[jid, fid] = 1 # object contact
                else:
                    contact_states[jid, fid] = 2 # ground contact
            elif pred==1:
                contact_states[jid, fid] = 0 # joint not in contact
            elif pred==2:
                contact_states[jid, fid] = 3 # joint occluded
            elif pred==3:
                contact_states[jid, fid] = -1 # undetected joint
            else:
                raise ValueError("Unknown pred: {0:d}".format(pred))

    return contact_states

def find_ground_contact_joints(contact_states):
    '''
    Given the contact_states array of size (njoints, nframes), this function
    returns a list of joints that have ever been in contact with the ground.
    '''
    njoints = contact_states.shape[0]
    joints_having_ground_contact = []
    ground_or_object = np.max(contact_states, axis=1)
    for j in range(njoints): # 24 joints
        # 0: no contact, 1: object contact, 2: ground contact
        if ground_or_object[j] == 2:
            joints_having_ground_contact.append(j)
    return joints_having_ground_contact


class ContactStates:

    def __init__(self, path_contact_states, video_name, action,
                 frame_start, frame_end):

        # ------------------------------------------------------------------
        print("(contact_states.py) Loading contact states from path ...\n"
          " - {0:s}".format(path_contact_states))

        # Mapping from the 9 joints with contact states recognized from image
        # to the joints in our human model
        # -----------------------------------------
        # | Joint name | ContactRec ID |  Ours ID |
        # | ---------- | ------------- | -------- |
        # | neck       |             0 |       12 |
        # | l_fingers  |             1 |       18 |
        # | r_fingers  |             2 |       23 |
        # | l_knee     |             3 |        2 |
        # | r_knee     |             4 |        6 |
        # | l_ankle    |             5 |        3 |
        # | r_ankle    |             6 |        7 |
        # | l_toes     |             7 |        4 |
        # | r_toes     |             8 |        8 |
        # -----------------------------------------
        joint_mapping = [12, 18, 23, 2, 6, 3, 7, 4, 8]

        contact_states = load_raw_contact_states(
            path_contact_states, video_name, joint_mapping,
            frame_start, frame_end)

        # ------------------------------------------------------------------
        print("(contact_states.py) Post-processing contact states ...")

        # Deal with joints recognized as occluded
        for jid in joint_mapping:
            for fid in range(contact_states.shape[1]):
                label = contact_states[jid, fid]
                if label==3:
                    if fid==0:
                        # Set to "not in contact" for the first frame
                        contact_states[jid, fid] = 0
                    else:
                        # Set to the contact state of the previous frame
                        contact_states[jid, fid] = contact_states[jid, fid-1]

        # Deal with the joints not detected by Openpose
        contact_states = self.fill_undetected_joints(
            contact_states,
            joint_mapping[0],   # neck id
            joint_mapping[1:3], # hand ids
            joint_mapping[3:5], # knee ids
            joint_mapping[5:7], # sole ids
            joint_mapping[7:9]) # toes ids

        # Remove toe contact if the foot sole is in contact
        contact_states = self.adjust_toe_contacts(
            contact_states,
            joint_mapping[5:7], # sole ids
            joint_mapping[7:9]) # toes ids

        # ------------------------------------------------------------------
        # Apply action-specific heuristics

        # Barbell action: remove single-hand contacts for heavy object
        if action=="barbell":
            for fid in range(contact_states.shape[1]):
                if contact_states[18,fid]*contact_states[23,fid] == 0:
                    contact_states[18,fid] = 0
                    contact_states[23,fid] = 0
        else:
            # Ignore neck contacts for non-barbell actions
            contact_states[12, :] = 0

        # Parkour actions: ignore knee contacts
        if action in ['kv', 'mu', 'pu', 'sv']:
            contact_states[2, :] = 0
            contact_states[6, :] = 0

        # ------------------------------------------------------------------
        # Generate misc info

        fixed_joints, sliding_joints = self.get_fixed_and_sliding_joints(
            action, joint_mapping)
        self.nj, self.nf = contact_states.shape[:2]
        self.contact_states = contact_states
        self.contact_types = self.generate_contact_types(
            contact_states, fixed_joints, sliding_joints)
        self.contact_mapping = self.get_joint_to_contact_frame_mapping()
        self.joints_having_ground_contact = find_ground_contact_joints(
            contact_states)


    def fill_undetected_joints(self, contact_states, neck_id, hand_ids,
                               knee_ids, sole_ids, toes_ids):
        '''
        Assign contact state for joints not detected by Openpose-video,
        i.e., entries with value -1 in the input array contact_states.
        We follow the following convention to "fill" those blanks:
        For example, if a person's left knee is undetected but right knee
        is detected and recognized as in contact (with ground), then we
        manually set the left knee to be in contact with the ground as well.
        The same principal applies to left/right hands, soles and toes.
        '''
        for i in range(contact_states.shape[1]):

            if contact_states[neck_id,i]==-1:
                contact_states[neck_id,i] = 0

            for j in [0, 1]:
                if contact_states[hand_ids[j],i]==-1 and \
                   contact_states[hand_ids[1-j],i]!=-1:
                    contact_states[hand_ids[j],i] = \
                        contact_states[hand_ids[1-j],i]

                if contact_states[knee_ids[j],i]==-1 and \
                   contact_states[knee_ids[1-j],i]!=-1:
                    contact_states[knee_ids[j],i] = \
                        contact_states[knee_ids[1-j],i]

                if contact_states[sole_ids[j],i]==-1 and \
                   contact_states[sole_ids[1-j],i]!=-1:
                    contact_states[sole_ids[j],i] = \
                        contact_states[sole_ids[1-j],i]

                if contact_states[toes_ids[j],i]==-1 and \
                   contact_states[toes_ids[1-j],i]!=-1:
                    contact_states[toes_ids[j],i] = \
                        contact_states[toes_ids[1-j],i]

        return contact_states


    def adjust_toe_contacts(self, contact_states, sole_ids, toes_ids):
        '''
        Modify contact_states such that, if left (right) sole is in contact,
        then the corresponding left (right) toes are not in contact.
        '''
        for i in range(contact_states.shape[1]):
            for j in [0, 1]:
                sole_id = sole_ids[j]
                toes_id = toes_ids[j]
                if contact_states[sole_id, i] in [1,2]:
                    contact_states[toes_id, i] = 0

        return contact_states


    def get_fixed_and_sliding_joints(self, action, joint_mapping):
        '''
        Generate the lists of fixed and of sliding joints
        '''
        sliding_joints = [4,8] # toes
        if action in ["hammer", "spade"]:
            sliding_joints.extend([18,23]) # hands

        fixed_joints = []
        for j in sorted(joint_mapping):
            if j not in sliding_joints:
                fixed_joints.append(j)

        return fixed_joints, sliding_joints


    def generate_contact_types(self,
            contact_states, fixed_joints, sliding_joints):
        '''
        Generate contact types based on contact_states and the lists of fixed
        and sliding joints. The function outputs a contact_types array which
        has the same size as contact_states. contact_types will help the
        trajectory estimator to distinguish between joints in fixed contact
        (label 1) and joints in sliding contact (label 2).
        '''
        contact_types = contact_states.copy()
        for i in range(contact_types.shape[1]):
            for j in fixed_joints:
                if contact_types[j, i] > 0:
                    contact_types[j, i] = 1
            for j in sliding_joints:
                if contact_types[j, i] > 0:
                    contact_types[j, i] = 2

        return contact_types


    def get_number_of_contact_frames(self, val):
        '''
        Get the total number of contact points (implemented as operational
        frames using Pinocchio) from contact_states, according to the input
        val:
        - 1 for object contact points
        - 2 for ground contact points
        '''
        num_contact_points = 0
        num_contact_joints = 0
        list_contact_joints = []
        for j in range(self.nj):
            if val in self.contact_states[j].tolist():
                # num_contact_joints counts 1 if j is in contact
                num_contact_joints += 1
                list_contact_joints.append(j)
                # num_contact_points counts 4 for ankle joints, 1 otherwise
                if val == 2 and j in [3, 7]:
                    num_contact_points = num_contact_points+4
                else:
                    num_contact_points = num_contact_points+1
        return num_contact_points, num_contact_joints, list_contact_joints


    def find_contact_mapping(self, contact_mapping_binary):
        '''
        Returns an 1D array whose entries are the indices of the first 1
        in each row of the 2D array contact_mapping_binary.
        '''
        nrows, ncols = contact_mapping_binary.shape
        contact_mapping = np.zeros(nrows).astype(int)
        for j in range(nrows):
            for k in range(ncols):
                # Save the index of the first 1 entry
                if contact_mapping_binary[j, k] == 1:
                    contact_mapping[j] = k
                    break
                # Set to -1 for rows with only zeros
                if k == ncols-1:
                    contact_mapping[j] = -1
        contact_mapping = contact_mapping+1
        return contact_mapping


    def get_joint_to_contact_frame_mapping(self):
        '''
        Generate a binary array contact_mapping indicating the indices of
        the contact points (operational frames in Pinocchio) for each person
        joint. We assume that a person joint cannot be in contact with object
        and ground in a single video sequence.
        '''
        num_object_contacts, num_object_contact_joints, list_object_contact_joints = self.get_number_of_contact_frames(1)
        num_ground_contacts, num_ground_contact_joints, list_ground_contact_joints = self.get_number_of_contact_frames(2)
        contact_mapping_binary = np.zeros(
            (self.nj, max(num_object_contacts, num_ground_contacts)))
        contact_mapping_binary = np.matrix(contact_mapping_binary.astype(int))
        for val in [1, 2]:
            fid = 0
            for j in range(self.nj):
                if val in self.contact_states[j].tolist():
                    if val == 2 and j in [3, 7]:
                        for k in range(4):
                            contact_mapping_binary[j, fid+k] = 1
                        fid = fid + 4
                    else:
                        contact_mapping_binary[j, fid] = 1
                        fid = fid + 1

        self.contact_mapping_binary = contact_mapping_binary
        self.num_ground_contacts = num_ground_contacts
        self.num_ground_contact_joints = num_ground_contact_joints
        self.list_ground_contact_joints = list_ground_contact_joints
        self.num_object_contacts = num_object_contacts
        self.num_object_contact_joints = num_object_contact_joints
        self.list_object_contact_joints = list_object_contact_joints
        return self.find_contact_mapping(contact_mapping_binary)
