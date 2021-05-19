"""*================================================================
*   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
*   
*   Filename    : swc_handler.py
*   Author      : Yufeng Liu
*   Date        : 2021-03-15
*   Description : 
*
================================================================*"""
from copy import deepcopy

def parse_swc(swc_file):
    tree = []
    with open(swc_file) as fp:
        fp.readline() # skip the header line
        for line in fp.readlines():
            line = line.strip()
            if not line: continue
            idx, type_, x, y, z, r, p = line.split()
            idx = int(idx)
            type_ = int(type_)
            x = float(x)
            y = float(y)
            z = float(z)
            r = float(r)
            p = int(p)
            tree.append((idx, type_, x, y, z, r, p))
    
    return tree

def write_swc(tree, swc_file):
    with open(swc_file, 'w') as fp:
        fp.write(f'##n type x y z r parent\n')
        for leaf in tree:
            idx, type_, x, y, z, r, p = leaf
            fp.write(f'{idx:d} {type_:d} {x:.2f} {y:.2f} {z:.2f} {r:.1f} {p:d}\n')
 
def find_soma_node(tree, p_soma=-1, p_idx_in_leaf=-1):
    for leaf in tree:
        if leaf[p_idx_in_leaf] == p_soma:
            #print('Soma: ', leaf)
            return leaf[0]
    raise ValueError("Could not find the soma node!")

def get_child_dict(tree, p_idx_in_leaf=-1):
    child_dict = {}
    for leaf in tree:
        p_idx = leaf[p_idx_in_leaf]
        if p_idx in child_dict:
            child_dict[p_idx].append(leaf[0])
        else:
            child_dict[p_idx] = [leaf[0]]
    return child_dict

def is_in_box(x, y, z, imgshape):
    """
    imgshape must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
        x > imgshape[2] - 1 or \
        y > imgshape[1] - 1 or \
        z > imgshape[0] - 1:
        return False
    return True

def trim_swc(tree_orig, imgshape, keep_candidate_points=True):
    """
    Trim the out-of-box and non_connecting leaves
    """

    def traverse_leaves(idx, child_dict, good_points, cand_pints, pos_dict):
        leaf = pos_dict[idx]
        p_idx, ib = leaf[-2:]

        if (p_idx in good_points) or (p_idx == -1):
            if ib:
                good_points.add(idx)    # current node
            else:
                cand_points.add(idx)
                return

        if idx not in child_dict:
            return

        for new_idx in child_dict[idx]:
            traverse_leaves(new_idx, child_dict, good_points, cand_pints, pos_dict)

    # execute trimming
    pos_dict = {}
    tree = deepcopy(tree_orig)
    for i, leaf in enumerate(tree_orig):
        idx, type_, x, y, z, r, p = leaf
        leaf = (idx, type_, x, y, z, r, p, is_in_box(x,y,z,imgshape))
        pos_dict[idx] = leaf
        tree[i] = leaf

    good_points = set() # points and all its upstream parents are in-box
    cand_points = set() # all upstream parents are in-box, itself not
    # initialize the visited set with soma, whose parent index is -1
    soma_idx = None
    for leaf in tree:
        if leaf[-2] == -1:
            soma_idx = leaf[0]
            break

    child_dict = {}
    for leaf in tree:
        if leaf[-2] in child_dict:
            child_dict[leaf[-2]].append(leaf[0])
        else:
            child_dict[leaf[-2]] = [leaf[0]]
    # do DFS searching
    #print(soma_idx)
    traverse_leaves(soma_idx, child_dict, good_points, cand_points, pos_dict)
    #print("#good/#cand/#total:", len(good_points), len(cand_points), len(pos_dict))  

    # return the tree, (NOTE: without order)
    tree_trim = []
    if keep_candidate_points:
        keep_points = good_points | cand_points
    else:
        keep_points = good_points

    for i, leaf in enumerate(tree):
        idx = leaf[0]
        if idx in keep_points:
            tree_trim.append(leaf[:-1])

    return tree_trim

