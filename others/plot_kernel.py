#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 23:05:13 2021

@author: lihao
"""
# this version implemented the predicted energy based on the joint dist of E*, E and F*.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import timeit
import logging
import warnings
#os.chdir('/home/a510396/testl')
os.chdir('./')
#os.chdir('/Users/HL/Desktop/Study/SFM')

from utils import io
#import hashlib
import numpy as np
import scipy as sp
import multiprocessing as mp
Pool = mp.get_context('fork').Pool

from solvers.analytic_train525 import Analytic
#from solvers.analytic_u_M15 import Analytic
#from solvers.analytic_u_pp import Analytic
from utils import perm
from utils.desc import Desc
from functools import partial


#global glob
#glob = {}

def _share_array(arr_np, typecode_or_type):
    """
    Return a ctypes array allocated from shared memory with data from a
    NumPy array.
    Parameters
    ----------
        arr_np : :obj:`numpy.ndarray`
            NumPy array.
        typecode_or_type : char or :obj:`ctype`
            Either a ctypes type or a one character typecode of the
            kind used by the Python array module.
    Returns
    -------
        array of :obj:`ctype`
    """

    arr = mp.RawArray(typecode_or_type, arr_np.ravel())
    return arr, arr_np.shape

#draw_strat_sample(dataset2['E'],100)

def _assemble_kernel_mat_wkr(
    j, tril_perms_lin, tril_perms_lin_mirror, sig, index_diff_atom,use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    r"""
    ----------
        j : int
            Index of training point.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array (int) containing all recovered permutations
            expanded as one large permutation to be applied to a tiled
            copy of the object to be permuted.
    Compute one row and column of the force field kernel matrix.
    The Hessian of the Matern kernel is used with n = 2 (twice
    differentiable). Each row and column consists of matrix-valued
    blocks, which encode the interaction of one training point with all
    others. The result is stored in shared memory (a global variable).
    """
    global glob

    R_desc_atom = np.frombuffer(glob['R_desc_atom']).reshape(glob['R_desc_shape_atom'])
    R_d_desc_atom = np.frombuffer(glob['R_d_desc_atom']).reshape(glob['R_d_desc_shape_atom'])
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    desc_func = glob['desc_func']
    
    n_train, dim_d = R_d_desc_atom.shape[:2]
    n_type=len(index_diff_atom)
    # 600; dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    #dim_ii =3 * n_atoms
     
    #n_perms =int(len(tril_perms_lin) / (dim_d * 6))
    n_perms = int(len(tril_perms_lin) / dim_d)  #12
    n_perm_atom=n_perms
    
    #blk_j = slice(j*3 , (j + 1) *3)
    #blk_j = slice(j * dim_i, (j + 1) * dim_i)

    blk_j = slice(j * dim_i, (j + 1) * dim_i)
       
    
    keep_idxs_3n = slice(None)  # same as [:]
    
    
    rj_desc_perms = np.reshape(
        np.tile(R_desc_atom[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
    # rj_desc_perms = 2 * 66
    
    rj_d_desc = desc_func.d_desc_from_comp(R_d_desc_atom[j, :, :])[0][
        :, keep_idxs_3n
    ]  # convert descriptor back to full representation
    # rj_d_desc 66 * 36
    
    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
    )
    #  rj_d_desc_perms 36 * 66 * 2
    
    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2
    
    dim_i_keep = rj_d_desc.shape[1]  # 36
    diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))   # 66 * 36
    #diff_ab_outer_perms_ij = np.empty((n_perm_atom,n_perm_atom,dim_d, dim_i_keep)) 
    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36
    # 66 * 36
   # diff_ab_perms = np.empty((n_perm_atom, dim_d))  # 12 * 66
    #ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros! # 1* 66 * 36
    #k = np.empty((dim_i, dim_i_keep))   # 36 * 36
    #k=np.empty((1))
    ri_d_desc = np.zeros((1, dim_d, dim_i))
    k = np.zeros((dim_i, dim_i_keep))
    #k = np.empty((dim_i, dim_i_keep))
    
    if use_E_cstr:
        ri_d_desc = np.zeros((1, dim_d, dim_i-1))
        k = np.zeros((dim_i-1, dim_i_keep))
        diff_ab_perms_t = np.empty((n_perm_atom, dim_d))

    #index_C=int(index_atom+n_atoms/2*3*index_type ) #2=n_type
    
    #for i in range(0, j+1):
    for i in range(0, j+1):
        

        blk_i = slice(i * dim_i, (i + 1) * dim_i)
            #blk_i_full = slice(i * dim_i, (i + 1) * dim_i-1)

        np.subtract(R_desc_atom[i, :], rj_desc_perms, out=diff_ab_perms)
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5
        
        np.einsum(
            'ki,kj->ij',
            diff_ab_perms * mat52_base_perms[:, None] * 5,
            np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
            out=diff_ab_outer_perms
        )
        
        diff_ab_outer_perms -= np.einsum(
            'ikj,j->ki',
            rj_d_desc_perms,
            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
        )
        
        desc_func.d_desc_from_comp(R_d_desc_atom[i, :, :], out=ri_d_desc)
        
        #np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)

        tem_d=np.empty((dim_d, 3)) 
        if(n_atoms<=12):
            np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
        else:
            #k1 = np.empty((3,3))
            for l in range(0, n_type):
                lenl=len(index_diff_atom[l])
                k1 = np.empty((3*lenl,3*lenl))
                index = np.tile(np.arange(3),lenl)+3*np.repeat(index_diff_atom[l],3)
                
                #index = np.arange(3)+3*l
                
                np.dot(ri_d_desc[0].T[index,:], diff_ab_outer_perms[:,index], out=k1)
                k[np.ix_(index,index)]=k1.copy()

        
        K[blk_i, blk_j] = -k
        K[blk_j, blk_i] = np.transpose(-k)
       
        
        
        
   
    return blk_j.stop - blk_j.start



def _assemble_kernel_mat_wkr_test(
    j,tril_perms_lin, tril_perms_lin_mirror, sig,index_diff_atom, use_E_cstr=False, exploit_sym=False, cols_m_limit=None
):
    r"""
    ----------
        j : int
            Index of training point.
        tril_perms_lin : :obj:`numpy.ndarray`
            1D array (int) containing all recovered permutations
            expanded as one large permutation to be applied to a tiled
            copy of the object to be permuted.
    Compute one row and column of the force field kernel matrix.
    The Hessian of the Matern kernel is used with n = 2 (twice
    differentiable). Each row and column consists of matrix-valued
    blocks, which encode the interaction of one training point with all
    others. The result is stored in shared memory (a global variable).
    """
    global glob

    R_desc = np.frombuffer(glob['R_desc']).reshape(glob['R_desc_shape'])
    R_d_desc = np.frombuffer(glob['R_d_desc']).reshape(glob['R_d_desc_shape'])

    R_desc_val = np.frombuffer(glob['R_desc_val']).reshape(glob['R_desc_shape_val'])
    R_d_desc_val = np.frombuffer(glob['R_d_desc_val']).reshape(glob['R_d_desc_shape_val'])    
    
    #glob['R_desc_val'], glob['R_desc_shape_val'] = _share_array(R_desc_val_atom, 'd')
    #glob['R_d_desc_val'], glob['R_d_desc_shape_val'] = _share_array(R_d_desc_val_atom, 'd')    
 
    K = np.frombuffer(glob['K']).reshape(glob['K_shape'])
    
    desc_func = glob['desc_func']
    n_type=len(index_diff_atom)
    
    n_train, dim_d = R_d_desc.shape[:2]
    n_val, dim_d = R_d_desc_val.shape[:2]
    # dim_d =66

    n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
    dim_i = 3 * n_atoms  # 36
    dim_ii =3 * n_atoms
    if use_E_cstr:
        dim_i=3 * n_atoms+1

    n_perms = int(len(tril_perms_lin) / dim_d)
    n_perm_atom=n_perms
    
        
    blk_j = slice(j*dim_i , (j + 1)*dim_i )
    if use_E_cstr:
       blk_j = slice(j * dim_i, (j + 1) * dim_i -1 )

    #blk_j = slice(j * dim_i, (j + 1) * dim_i)
    keep_idxs_3n = slice(None)  # same as [:]
  
    rj_desc_perms = np.reshape(
        np.tile(R_desc[j, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
    # rj_desc_perms = 12 * 66
    
    rj_d_desc = desc_func.d_desc_from_comp(R_d_desc[j, :, :])[0][
        :, keep_idxs_3n
    ]  # convert descriptor back to full representation
    # rj_d_desc 66 * 36
    

    
    rj_d_desc_perms = np.reshape(
        np.tile(rj_d_desc.T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
    )
    #  rj_d_desc_perms 36 * 66 * 12
    
    mat52_base_div = 3 * sig ** 4
    sqrt5 = np.sqrt(5.0)
    sig_pow2 = sig ** 2
    
    dim_i_keep = rj_d_desc.shape[1]  # 36
    diff_ab_outer_perms = np.empty((dim_d, dim_i_keep))   # 66 * 36
    #diff_ab_outer_perms_ij = np.empty((n_perm_atom,n_perm_atom,dim_d, dim_i_keep)) 
    diff_ab_perms = np.empty((n_perm_atom, dim_d))# 66 * 36
    diff_ab_perms_t = np.empty((n_perm_atom, dim_d))
   # diff_ab_perms = np.empty((n_perm_atom, dim_d))  # 12 * 66
    #ri_d_desc = np.zeros((1, dim_d, dim_i)) # must be zeros! # 1* 66 * 36
    #k = np.empty((dim_i, dim_i_keep))   # 36 * 36
    #k=np.empty((1))
    ri_d_desc = np.zeros((1, dim_d, dim_i))
    k = np.zeros((dim_i, dim_i_keep))
    k1 = np.zeros((3, 3))
    if use_E_cstr:
        ri_d_desc = np.zeros((1, dim_d, dim_i-1))
        k = np.zeros((dim_i-1, dim_i_keep))
    #index_C=int(index_atom+n_atoms/2*3*index_type ) #2=n_type
    
    for i in range(0, n_val):
        blk_i = slice(i * dim_i, (i + 1) * dim_i)
        
        if use_E_cstr:
            blk_i = slice(i * dim_i, (i + 1) * dim_i-1)
            blk_i_full = slice(i * dim_i, (i + 1) * dim_i-1)

        #R_desc_val
        np.subtract(R_desc_val[i, :], rj_desc_perms, out=diff_ab_perms)
        norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
        mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5
        
        np.einsum(
            'ki,kj->ij',
            diff_ab_perms * mat52_base_perms[:, None] * 5,
            np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
            out=diff_ab_outer_perms
        )
        
        diff_ab_outer_perms -= np.einsum(
            'ikj,j->ki',
            rj_d_desc_perms,
            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
        )
        
        desc_func.d_desc_from_comp(R_d_desc_val[i, :, :], out=ri_d_desc)
        #k1 = np.empty((3,3))
        for l in range(0, n_type):
            k1 = np.empty((3*len(index_diff_atom[l]),3*len(index_diff_atom[l])))
            index = np.tile(np.arange(3),len(index_diff_atom[l]))+3*np.repeat(index_diff_atom[l],3)
            
            #index = np.arange(3)+3*l
            
            np.dot(ri_d_desc[0].T[index,:], diff_ab_outer_perms[:,index], out=k1)
            k[np.ix_(index,index)]=k1.copy()
        
        
        
        if use_E_cstr:
            ri_desc_perms = np.reshape(
        np.tile(R_desc_val[i, :], n_perm_atom)[tril_perms_lin_mirror], (n_perm_atom, -1), order='F'
    )
            ri_d_desc_perms = np.reshape(
            np.tile(ri_d_desc[0].T, n_perm_atom)[:, tril_perms_lin_mirror], (-1, dim_d, n_perm_atom)
        )
            #diff_ab_perms = R_desc_atom[i, :] - rj_desc_perms
            #norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)

            K_fe = (
                5
                * diff_ab_perms
                / (3 * sig ** 3)
                * (norm_ab_perms[:, None] + sig)
                * np.exp(-norm_ab_perms / sig)[:, None]
            )
            K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
            #K[blk_i_full, (j + 1) * dim_i-1] = K_fe  # vertical
            #K[blk_j, (i + 1) * dim_i-1] = K_fe 
            
            np.subtract(R_desc[j, :], ri_desc_perms, out=diff_ab_perms_t)
            norm_ab_perms_t = sqrt5 * np.linalg.norm(diff_ab_perms_t, axis=1)
            
            K_fet = (
                5
                * diff_ab_perms_t
                / (3 * sig ** 3)
                * (norm_ab_perms_t[:, None] + sig)
                * np.exp(-norm_ab_perms_t / sig)[:, None]
            )
            
            # K_fet = (
            #     5
            #     * diff_ab_perms
            #     / (3 * sig ** 3)
            #     * (norm_ab_perms[:, None] + sig)
            #     * np.exp(-norm_ab_perms / sig)[:, None]
            # )
            K_fet = np.einsum('ik,jki -> j', K_fet, ri_d_desc_perms)
            # K_fet = np.einsum('ik,kj -> ij', K_fet, ri_d_desc[0])
            # K_fet = np.sum(K_fet,0)

            K[blk_i_full, (j + 1) * dim_i-1] = K_fet
            K[(i + 1) * dim_i-1, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal
            #K[(i + 1) * dim_i-1, blk_j]
            
            #K[(i + 1) * dim_i-1, blk_j] = K_fe[keep_idxs_3n]  # lower horizontal

            # K[(i + 1) * dim_i-1, (j + 1) * dim_i-1] = (
            #     1 + (norm_ab_perms_t / sig) * (1 + norm_ab_perms_t / (3 * sig))
            # ).dot(np.exp(-norm_ab_perms_t / sig))
            
            # K[(i + 1) * dim_i-1, (j + 1) * dim_i-1] = (
            #     1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            # ).dot(np.exp(-norm_ab_perms / sig))
            
            K[(i + 1) * dim_i-1, (j + 1) * dim_i-1]  = (
            1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
        ).dot(np.exp(-norm_ab_perms / sig)) 
        
        K[blk_i, blk_j] = -k
       # -----------
    
  
    

        
    return blk_j.stop - blk_j.start


class GDMLTrain(object):
    def __init__(self, max_processes=None, use_torch=False):
        global glob
        if 'glob' not in globals():  # Don't allow more than one instance of this class.
            glob = {}
        else:
            raise Exception(
                'You can not create multiple instances of this class. Please reuse your first one.'
            )

        self.log = logging.getLogger(__name__)

        self._max_processes = max_processes
        self._use_torch = use_torch

    def __del__(self):

        global glob

        if 'glob' in globals():
            del glob
            

    def draw_strat_sample(self,T,n, excl_idxs=None):
            """Draw sample from dataset that preserves its original distribution.
            The distribution is estimated from a histogram were the bin size is
            determined using the Freedman-Diaconis rule. This rule is designed to
            minimize the difference between the area under the empirical
            probability distribution and the area under the theoretical
            probability distribution. A reduced histogram is then constructed by
            sampling uniformly in each bin. It is intended to populate all bins
            with at least one sample in the reduced histogram, even for small
            training sizes.
            Parameters
            ----------
                T : :obj:`numpy.ndarray`
                    Dataset to sample from.
                n : int
                    Number of examples.
                excl_idxs : :obj:`numpy.ndarray`, optional
                    Array of indices to exclude from sample.
            Returns
            -------
                :obj:`numpy.ndarray`
                    Array of indices that form the sample.
            """
            if excl_idxs is None or len(excl_idxs) == 0:
                excl_idxs = None
    
            if n == 0:
                return np.array([], dtype=np.uint)
    
            if T.size == n:  # TODO: this only works if excl_idxs=None
                assert excl_idxs is None
                return np.arange(n)
    
            if n == 1:
                idxs_all_non_excl = np.setdiff1d(
                    np.arange(T.size), excl_idxs, assume_unique=True
                )
                return np.array([np.random.choice(idxs_all_non_excl)])
    
            # Freedman-Diaconis rule
            h = 2 * np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
            n_bins = int(np.ceil((np.max(T) - np.min(T)) / h)) if h > 0 else 1
            n_bins = min(
                n_bins, int(n / 2)
            )  # Limit number of bins to half of requested subset size.
    
            bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
            idxs = np.digitize(T, bins)
    
            # Exclude restricted indices.
            if excl_idxs is not None and excl_idxs.size > 0:
                idxs[excl_idxs] = n_bins + 1  # Impossible bin.
    
            uniq_all, cnts_all = np.unique(idxs, return_counts=True)
    
            # Remove restricted bin.
            if excl_idxs is not None and excl_idxs.size > 0:
                excl_bin_idx = np.where(uniq_all == n_bins + 1)
                cnts_all = np.delete(cnts_all, excl_bin_idx)
                uniq_all = np.delete(uniq_all, excl_bin_idx)
    
            # Compute reduced bin counts.
            reduced_cnts = np.ceil(cnts_all / np.sum(cnts_all, dtype=float) * n).astype(int)
            reduced_cnts = np.minimum(
                reduced_cnts, cnts_all
            )  # limit reduced_cnts to what is available in cnts_all
    
            # Reduce/increase bin counts to desired total number of points.
            reduced_cnts_delta = n - np.sum(reduced_cnts)
    
            while np.abs(reduced_cnts_delta) > 0:
    
                # How many members can we remove from an arbitrary bucket, without any bucket with more than one member going to zero?
                max_bin_reduction = np.min(reduced_cnts[np.where(reduced_cnts > 1)]) - 1
    
                # Generate additional bin members to fill up/drain bucket counts of subset. This array contains (repeated) bucket IDs.
                outstanding = np.random.choice(
                    uniq_all,
                    min(max_bin_reduction, np.abs(reduced_cnts_delta)),
                    p=(reduced_cnts - 1) / np.sum(reduced_cnts - 1, dtype=float),
                    replace=True,
                )
                uniq_outstanding, cnts_outstanding = np.unique(
                    outstanding, return_counts=True
                )  # Aggregate bucket IDs.
    
                outstanding_bucket_idx = np.where(
                    np.in1d(uniq_all, uniq_outstanding, assume_unique=True)
                )[
                    0
                ]  # Bucket IDs to Idxs.
                reduced_cnts[outstanding_bucket_idx] += (
                    np.sign(reduced_cnts_delta) * cnts_outstanding
                )
                reduced_cnts_delta = n - np.sum(reduced_cnts)
    
            # Draw examples for each bin.
            idxs_train = np.empty((0,), dtype=int)
            for uniq_idx, bin_cnt in zip(uniq_all, reduced_cnts):
                idx_in_bin_all = np.where(idxs.ravel() == uniq_idx)[0]
                idxs_train = np.append(
                    idxs_train, np.random.choice(idx_in_bin_all, bin_cnt, replace=False)
                )
    
            return idxs_train

    
    def _assemble_kernel_mat(
            self,
            index_diff_atom,
            R_desc,
            R_d_desc,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,  # TODO: document me
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
    
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
    
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_train*3 #* 6   
        K_n_rows = n_train * dim_i  
        K_n_cols = n_train * dim_i   
        
        # if use_E_cstr:
        #     K_n_rows += n_train
        #     K_n_cols += n_train
        #K_n_cols = n_train*3 # * 6  
        #K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        exploit_sym = False
        cols_m_limit = None
        is_M_subset = (
                isinstance(col_idxs, slice)
                and (col_idxs.start is None or col_idxs.start % dim_i == 0)
                and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
                and col_idxs.step is None
            )
        if is_M_subset:
            M_slice_start = (None if col_idxs.start is None else int(col_idxs.start / dim_i))
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)
    
            J = range(*M_slice.indices(n_train))
    
            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop
                
        #K = mp.RawArray('d', n_type * K_n_rows * K_n_cols)
        K = mp.RawArray('d',  K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, (K_n_rows, K_n_cols)
        glob['R_desc_atom'], glob['R_desc_shape_atom'] = _share_array(R_desc, 'd')
        glob['R_d_desc_atom'], glob['R_d_desc_shape_atom'] = _share_array(R_d_desc, 'd')
    
        glob['desc_func'] = desc
        start = timeit.default_timer()
        pool = Pool(mp.cpu_count())
        #pool = Pool(self._max_processes)
        todo, done = K_n_cols, 0
        
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                index_diff_atom=index_diff_atom,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        ):
            done += done_wkr
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()
        dur_s = (stop - start)
        
        glob.pop('K', None)
        glob.pop('R_desc_atom', None)
        glob.pop('R_d_desc_atom', None)
    
        return np.frombuffer(K).reshape(glob['K_shape']),dur_s
    
        
    
    def create_task(self,
            train_dataset,
            n_train,
            valid_dataset,
            n_valid,
            n_test,
            sig,
            lam=1e-15,
            batch_size=1,
            use_sym=True,
            use_E=True,
            use_E_cstr=False,
            use_cprsn=False,
            solver='analytic',  # TODO: document me
            solver_tol=1e-4,  # TODO: document me
            n_inducing_pts_init=25,  # TODO: document me
            interact_cut_off=None,  # TODO: document me
            callback=None,  # TODO: document me
        ):
        n_atoms = train_dataset['R'].shape[1]
        md5_train = io.dataset_md5(train_dataset)
        md5_valid = io.dataset_md5(valid_dataset)
        
        
        #if 'E' in train_dataset:
        idxs_train = self.draw_strat_sample(
                    train_dataset['E'], n_train
                )
        
        excl_idxs = (
                idxs_train if md5_train == md5_valid else np.array([], dtype=np.uint)
            )
        
        idxs_valid = self.draw_strat_sample(
                    valid_dataset['E'], n_valid, excl_idxs=excl_idxs,
                )
        
        excl_idxs1 = np.concatenate((idxs_train, idxs_valid))
        
        idxs_test = self.draw_strat_sample(
                    valid_dataset['E'], n_test, excl_idxs=excl_idxs1,
                )
        #else:
    # =============================================================================
    #             idxs_train = np.random.choice(
    #                 np.arange(train_dataset['F'].shape[0]),
    #                 n_train - m0_n_train,
    #                 replace=False,
    #             )
    # =============================================================================
        R_train = train_dataset['R'][idxs_train, :, :]
        R_val = train_dataset['R'][idxs_valid, :, :]
        R_test = train_dataset['R'][idxs_test, :, :]
        task = {
            'type': 't',
            #'code_version': __version__,
            'dataset_name': train_dataset['name'].astype(str),
            'dataset_theory': train_dataset['theory'].astype(str),
            'z': train_dataset['z'],
            'R_train': R_train,
            'R_val': R_val,
            'R_test':R_test,
            'F_train': train_dataset['F'][idxs_train, :, :],
            'F_val': train_dataset['F'][idxs_valid, :, :],
            'F_test': train_dataset['F'][idxs_test, :, :],
            'idxs_train': idxs_train,
            'md5_train': md5_train,
            'idxs_valid': idxs_valid,
            'idex_test':idxs_test,
            'md5_valid': md5_valid,
            'sig': sig,
            'lam': lam,
            'batch_size':batch_size,
            'use_E': use_E,
            'use_E_cstr': use_E_cstr,
            'use_sym': use_sym,
            'use_cprsn': use_cprsn,
            'solver_name': solver,
            'solver_tol': solver_tol,
            'n_inducing_pts_init': n_inducing_pts_init,
            'interact_cut_off': interact_cut_off,
        }
        
        if use_E:
                task['E_train'] = train_dataset['E'][idxs_train]
                task['E_val'] = train_dataset['E'][idxs_valid]
                task['E_test'] = train_dataset['E'][idxs_test]
        
        if use_sym:
                n_train = R_train.shape[0]
                R_train_sync_mat = R_train
                if n_train > 1000:
                    R_train_sync_mat = R_train[
                        np.random.choice(n_train, 1000, replace=False), :, :
                    ]
    # =============================================================================
    #                 self.log.info(
    #                     'Symmetry search has been restricted to a random subset of 1000/{:d} training points for faster convergence.'.format(
    #                         n_train
    #                     )
    # =============================================================================
                    
    # sort the permutation matrix by first column, then the mirror-wise symmetries are together
                nb = perm.find_perms(
                            R_train_sync_mat,
                            #train_dataset['R'][range(500),:,:],
                            train_dataset['z'],
                            lat_and_inv=None,
                            callback=callback,
                            max_processes=None,
                        )
                task['perms']=nb[nb[:,0].argsort()]
                #a[a[:,0].argsort()]
        else:
                task['perms'] = np.arange(train_dataset['R'].shape[1])[
                    None, :
                ]
        #task['F_train_atom']=train_dataset['F'][idxs_train, :, :]
        n_type,index_diff_atom=self.find_type(task['perms'])
        task['n_type']=n_type
        task['index_diff_atom']=index_diff_atom
        n_perms = task['perms'].shape[0]
        if use_cprsn and n_perms > 1:
    
                _, cprsn_keep_idxs = np.unique(
                    np.sort(task['perms'], axis=0), axis=1, return_index=True
                )
    
                task['cprsn_keep_atoms_idxs'] = cprsn_keep_idxs
    
        return task
    
    def find_type(self,task_perm):
        # to find out the number of different geometric atoms we have from this molecule
        # and the index of same type of atoms 
        n_perms=task_perm.shape[0]
        n_atoms=task_perm.shape[1]
        all_list=list(range(0,n_atoms))
        index_diff_atom=[]
        while len(all_list):
            index_atoms=np.unique(np.where(task_perm==all_list[0])[1])
            index_diff_atom.append(list(index_atoms))
            all_list=[ele for ele in all_list if ele not in index_atoms]
            #all_list.remove(list(index_atoms))
        n_type=len(index_diff_atom)
        return n_type,index_diff_atom
    
    def create_model(
        self,
        task,
        solver,
        R_desc_atom, #R_desc_atom
        R_d_desc_atom, #R_desc_atom_atom
        tril_perms_lin, # tril_perms_lin_mirror
        std,
        alphas_F,
        alphas_E=None,
        solver_resid=None,
        solver_iters=None,
        norm_y_train=None,
        inducing_pts_idxs=None,  # NEW : which columns were used to construct nystrom preconditioner
    ):
        n_train, dim_d = R_desc_atom.shape[:2]
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        
        i, j = np.tril_indices(n_atoms, k=-1)
        alphas_F_exp = alphas_F.reshape(-1, n_atoms, 3)
        
    
    def train(self, task,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
    
            #Train a model based on a training task.
    
        task = dict(task)
        solver = task['solver_name']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val, n_atoms = task['R_val'].shape[:2]
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        n_type = task['n_type']
        index_diff_atom = task['index_diff_atom']
        
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        tril_pos=task['perms']
        
        # tril_perms stores the 12 permutations on the 66 descriptor
        dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]

        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
    
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        R_val=task['R_val'] #.reshape(n_val,-1)
        # R is a n_train * 36 matrix        
        tril_perms_lin_mirror = tril_perms_lin
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        
        R_atom=R
        R_val_atom=R_val
        #R_mirror=
        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        F_train_atom=[]
        # if task['use_E_cstr']:
        #     #F_train_atom=np.empty((int(n_train*(n_atoms/n_type*3+1)),n_type))
            
        # else:
        #     F_train_atom=np.empty((int(n_train*n_atoms/n_type*3),n_type))
        F_val_atom=[]
            #F_val_atom=task['F_val'].ravel().copy()
        
        E_train = task['E_train'].ravel().copy()
        E_val = task['E_val'].ravel().copy()
        
        for i in range(n_type):
            index=np.array(index_diff_atom[i])

            F_train_atom.append(task['F_train'][:,index,:].reshape(int(n_train*(len(index_diff_atom[i])*3)),order='C'))
            F_val_atom.append(task['F_val'][:,index,:].reshape(int(n_val*(len(index_diff_atom[i])*3)),order='C'))

        #F_train_atom=task['F_train'].ravel().copy()
        #F_val_atom=task['F_val'].ravel().copy()
        
        #E_train_mean = None
        #y = task['F_train'].ravel().copy()
        #y_val= F_val_atom.copy()
        
        ye_val=E_val #- E_val_mean

        #y_std = np.std(y_atom)
        #y_atom /= y_std
        
        
        #sig_candid=np.arange(100,250,40)
        #sig_candid=np.arange(100,300,30) #alkane
        sig_candid=np.arange(10,20,10) #naphthalene_dft
        #sig_candid1=sig_candid
        sig_candid1=np.arange(6,35,3)#naphthalene_dft
        #sig_candid1=np.arange(6,15,1)  # this is for energy prediction of uracil
        #sig_candid1=np.arange(6,15,1)  # this is for energy prediction of aspirn
        num_i=sig_candid.shape[0]
        MSA=np.ones((num_i))*1e8
        #RMSE=np.empty((num_i))*1e8
        
        for i in range(num_i):
            y_atom= F_train_atom.copy()
            
            
            print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
            
            if solver == 'analytic':
                #gdml_train: analytic = Analytic(gdml_train, desc, callback=None)
        
                analytic = Analytic(self, desc, callback=callback)
                #alphas = analytic.solve(task, R_desc, R_d_desc, tril_perms_lin, tril_perms_lin_mirror, y)
                alphas, inverse_time, kernel_time = analytic.solve_xyz(task,sig_candid[i], R_desc_atom, R_d_desc_atom, tril_perms_lin, tril_perms_lin_mirror, y_atom)
                print('Kernel Processing time: '+str(kernel_time)+'seconds')
                print('training time:   '+str(inverse_time)+'seconds')
            
            # if task['use_E_cstr']:
            #     alphas_E = alphas[-n_train:]
            #     alphas_F = alphas[:-n_train]
            F_hat_val=[]
            F_hat_val_F=[]
            F_hat_val_E=[]
            # if task['use_E_cstr']:
            #     F_hat_val=np.empty((int(n_val*(n_atoms/n_type*3+1)),n_type)) 
            #     F_hat_val_F=np.empty((int(n_val*n_atoms/n_type*3),n_type)) 
            #     F_hat_val_E=np.empty((int(n_val),n_type))
            # else:
            #     F_hat_val=np.empty((int(n_val*(n_atoms/n_type*3)),n_type))
            #     F_hat_val_F=np.empty((int(n_val*n_atoms/n_type*3),n_type)) 
            
            
            K_r_all = self._assemble_kernel_mat_test(
                    index_diff_atom,
                    R_desc_atom,
                    R_d_desc_atom,
                    R_desc_atom,
                    R_d_desc_atom,
                    tril_perms_lin,
                    tril_perms_lin_mirror,
                    sig_candid[i],
                    desc,
                    use_E_cstr=False,
                    col_idxs= np.s_[:],
                    callback=None,
                )
            
            F_star=np.empty([n_val*3*n_atoms])
            for ind_i in range(n_type):
                index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)


                    
                index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
                index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)

                    # index_x=np.repeat(np.arange(n_val)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_val)
                    # index_y=np.repeat(np.arange(n_train)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_train)
                
                K_r=K_r_all[np.ix_(index_y,index_y)]
                #F_hat_val_i=np.matmul(K_r,np.array(alphas[ind_i]))
                #index_i=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)


               # F_hat_val_F.append(F_hat_val_i)
                #F_star[index_i]=F_hat_val_i
                
        return K_r_all,K_r_all[np.ix_(index_y,index_y)]
            
            #F_hat_val=np.matmul(K_r_all,alphas)
            

            #F_hat_val_E=(-np.matmul(K_r_all_e,alphas))-c
            # F_hat_val = F_hat_val1*y_std
            
                
                
        
    
    def test(self, task,sig_optim,sig_candid1_opt,alphas_opt,
            cprsn_callback=None,
            save_progr_callback=None,  # TODO: document me
            callback=None):
        task = dict(task)
        solver = task['solver_name']
        batch_size=task['batch_size']
        n_train, n_atoms = task['R_train'].shape[:2]
        n_val, n_atoms = task['R_test'].shape[:2]
        desc = Desc(
                n_atoms,
                interact_cut_off=task['interact_cut_off'],
                max_processes=None,
            )
        n_perms = task['perms'].shape[0]  # 12 on benzene
        tril_perms = np.array([desc.perm(p) for p in task['perms']])

        #tril_pos=task['perms']
        index_diff_atom = task['index_diff_atom']
        # tril_perms stores the 12 permutations on the 66 descriptor
        dim_i = 3 * n_atoms #36
        dim_d = desc.dim  #66 on benzene
        perm_offsets = np.arange(n_perms)[:, None] * dim_d
        # perm_offsets a [12,1] matrix stores the [0, 66, 66*2, ..., 12*66]
        tril_perms_lin = (tril_perms + perm_offsets).flatten('F')
        
          # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        n_type=task['n_type']
        
        lat_and_inv = None
        R = task['R_train']  #.reshape(n_train, -1) 
        
        R_val=task['R_test'] #.reshape(n_val,-1)
        # R is a n_train * 36 matrix 
        tril_perms_lin_mirror = tril_perms_lin
        # tril_perms_lin stores a vectorized permuations of all 12 permuations' descriptor position
        
        R_atom=R
        R_val_atom=R_val
        #R_mirror=
        R_desc_atom, R_d_desc_atom = desc.from_R(R_atom,lat_and_inv=lat_and_inv,
                callback=None)
        
        R_desc_val_atom, R_d_desc_val_atom = desc.from_R(R_val_atom,lat_and_inv=lat_and_inv,
                callback=None)
        F_train_atom=[]
        # if task['use_E_cstr']:
        #     #F_train_atom=np.empty((int(n_train*(n_atoms/n_type*3+1)),n_type))
            
        # else:
        #     F_train_atom=np.empty((int(n_train*n_atoms/n_type*3),n_type))
        F_val_atom=[]
            #F_val_atom=task['F_val'].ravel().copy()
        
        E_train = task['E_train'].ravel().copy()
        E_val = task['E_test'].ravel().copy()
        
        for i in range(n_type):
            index=np.array(index_diff_atom[i])

            F_train_atom.append(task['F_train'][:,index,:].reshape(int(n_train*(len(index_diff_atom[i])*3)),order='C'))
            F_val_atom.append(task['F_test'][:,index,:].reshape(int(n_val*(len(index_diff_atom[i])*3)),order='C'))
        ye_val=E_val
        
        #y_atom= F_train_atom.copy()
            
            
        print('This is tesing task : sigma='+repr(sig_optim))
        alphas=alphas_opt
        
        
        # if task['use_E_cstr']:
        #     alphas_E = alphas[-n_train:]
        #     alphas_F = alphas[:-n_train]
        #F_hat_val=[]
        F_hat_val_F=[]
        #F_hat_val_E=[]

        K_r_all = self._assemble_kernel_mat_test(
                index_diff_atom,
                R_desc_atom,
                R_d_desc_atom,
                R_desc_val_atom,
                R_d_desc_val_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_optim,
                desc,
                use_E_cstr=False,
                col_idxs= np.s_[:],
                callback=None,
            )
        
        F_star=np.empty([n_val*3*n_atoms])
        for ind_i in range(n_type):
            index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)


                
            index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
            index_y=np.repeat(np.arange(n_train)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)

                # index_x=np.repeat(np.arange(n_val)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_val)
                # index_y=np.repeat(np.arange(n_train)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_train)
            
            K_r=K_r_all[np.ix_(index_x,index_y)]
            F_hat_val_i=np.matmul(K_r,np.array(alphas[ind_i]))
            index_i=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)


            F_hat_val_F.append(F_hat_val_i)
            F_star[index_i]=F_hat_val_i
        ae=np.mean(np.abs(  np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom)))
        RMSE_F=np.sqrt(np.mean((np.concatenate(F_hat_val_F)-np.concatenate(F_val_atom))**2))/np.std(np.concatenate(F_val_atom))
        print(' This is the  MAE of F='+repr(ae)) 
        print(' This is the  RMSE of F='+repr(RMSE_F)) 
        #ae=np.mean(np.abs(F_hat_val_F-F_val_atom))
        MAE=ae
        
        if task['use_E_cstr']:
            lam_f=task['lam']
            lam_e=task['lam']
            print('   Starting training for energy:    ') 
            MSA_E_arr=[]
            start = timeit.default_timer()
        
                #print('This is '+repr(i)+'th task: sigma='+repr(sig_candid[i]))
            F_hat_val_E_ave=self._assemble_kernel_mat_Energy(
                index_diff_atom,
                R_desc_atom,
                R_d_desc_atom,
                R_desc_val_atom,
                R_d_desc_val_atom,
                tril_perms_lin,
                tril_perms_lin_mirror,
                sig_candid1_opt,
                lam_e,
                lam_f,
                batch_size,
                desc,
                F_star,#F_star_opt,
                E_train,
                use_E_cstr=task['use_E_cstr'],
                col_idxs= np.s_[:],
                callback=None,
            )
            stop = timeit.default_timer()
            dur_s = (stop - start)
            MSA_E=np.mean(np.abs(-F_hat_val_E_ave-ye_val))
            RMSE_E=np.sqrt(np.mean((-F_hat_val_E_ave-ye_val)**2))/np.std(ye_val)
            MSA_E_arr.append(MSA_E)
            print(' This is the testing task: It takes'+repr(dur_s)+'second; each estimation takes'+repr(dur_s/n_val)+' seconds') 
            print(' This is the testing task: MAE of E='+repr(MSA_E)) 
            print(' This is the testing task: RMSE of E='+repr(RMSE_E)) 
        
        
        
        
       
        return MAE

  
        
    
    def _assemble_kernel_mat_test(
            self,
            index_diff_atom,
            R_desc,
            R_d_desc,
            R_desc_val_atom,
            R_d_desc_val_atom,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            desc,  # TODO: document me
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
        n_val, dim_d = R_d_desc_val_atom.shape[:2]
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        #n_train , dim_d 66
        dim_i = 3 * int((1 + np.sqrt(8 * dim_d + 1)) / 2)  # dim = 3 * 12
    
        # Determine size of kernel matrix.
        #  **** need change
        #K_n_rows = n_val *3#* 6   
        K_n_rows = n_val * dim_i   
        K_n_cols = n_train * dim_i  # * 6  
        
        if use_E_cstr:
            K_n_rows += n_val
            K_n_cols += n_train
        #K_n_cols = len(range(*col_idxs.indices(K_n_rows)))
        exploit_sym = False
        cols_m_limit = None
        is_M_subset = (
                isinstance(col_idxs, slice)
                and (col_idxs.start is None or col_idxs.start % dim_i == 0)
                and (col_idxs.stop is None or col_idxs.stop % dim_i == 0)
                and col_idxs.step is None
            )
        if is_M_subset:
            M_slice_start = (None if col_idxs.start is None else int(col_idxs.start / dim_i))
            M_slice_stop = None if col_idxs.stop is None else int(col_idxs.stop / dim_i)
            M_slice = slice(M_slice_start, M_slice_stop)
    
            #J = range(*M_slice.indices(n_train))
            J = range(*M_slice.indices(n_train))
            if M_slice_start is None:
                exploit_sym = True
                cols_m_limit = M_slice_stop
                
        K = mp.RawArray('d',   K_n_rows * K_n_cols)
        glob['K'], glob['K_shape'] = K, ( K_n_rows, K_n_cols)
        glob['R_desc'], glob['R_desc_shape'] = _share_array(R_desc, 'd')
        glob['R_d_desc'], glob['R_d_desc_shape'] = _share_array(R_d_desc, 'd')
        glob['R_desc_val'], glob['R_desc_shape_val'] = _share_array(R_desc_val_atom, 'd')
        glob['R_d_desc_val'], glob['R_d_desc_shape_val'] = _share_array(R_d_desc_val_atom, 'd')    
    
        glob['desc_func'] = desc
        start = timeit.default_timer()
        pool = Pool(None)
        #pool = Pool(self._max_processes)
        todo, done = K_n_cols, 0
        
        for done_wkr in pool.imap_unordered(
            partial(
                _assemble_kernel_mat_wkr_test,
                tril_perms_lin=tril_perms_lin,
                tril_perms_lin_mirror=tril_perms_lin_mirror,
                sig=sig,
                index_diff_atom=index_diff_atom,
                use_E_cstr=use_E_cstr,
                exploit_sym=exploit_sym,
                cols_m_limit=cols_m_limit,
            ),
            J,
        ):
            done += done_wkr
            
        pool.close()
        pool.join()  # Wait for the worker processes to terminate (to measure total runtime correctly).
        stop = timeit.default_timer()
        
        glob.pop('K', None)
        glob.pop('R_desc', None)
        glob.pop('R_d_desc', None)
        glob.pop('R_desc_val', None)
        glob.pop('R_d_desc_val', None)
    
        return np.frombuffer(K).reshape(glob['K_shape'])
    
    def _assemble_kernel_mat_Energy(
            self,
            index_diff_atom,
            R_desc,
            R_d_desc,
            R_desc_val_atom,
            R_d_desc_val_atom,
            tril_perms_lin,
            tril_perms_lin_mirror,
            sig,
            lam_e,
            lam_f,
            batch_size,
            desc,  # TODO: document me
            F_star,
            E_train,
            use_E_cstr=False,
            col_idxs=np.s_[:],  # TODO: document me
            callback=None,
        ):
        r"""
            Compute force field kernel matrix.
        """
        #global glob
    
            # Note: This function does not support unsorted (ascending) index arrays.
            # if not isinstance(col_idxs, slice):
            #    assert np.array_equal(col_idxs, np.sort(col_idxs))
        #batch_size=int(10)
        n_val, dim_d = R_d_desc_val_atom.shape[:2]
        n_train, dim_d = R_d_desc.shape[:2]  #R_d_desc.shape (n_train, 66, 3)
        n_total=n_val+n_train
        n_atoms = int((1 + np.sqrt(8 * dim_d + 1)) / 2)
        dim_i = 3 * n_atoms
        #n_train , dim_d 66
        n_perms = int(len(tril_perms_lin) / dim_d)
        E_predict = np.empty(n_val)
        
        keep_idxs_3n = slice(None)
        
        K_EE_all = np.empty((n_val+n_train,n_val+n_train))
        K_EF_all = np.empty((n_val+n_train,n_val*3*n_atoms))
        K_FE_all = np.empty((n_val*3*n_atoms,n_val+n_train))
        
        K_FsFs_all = np.empty((n_val*3*n_atoms,n_val*3*n_atoms))
        
        mat52_base_div = 3 * sig ** 4
        sqrt5 = np.sqrt(5.0)
        sig_pow2 = sig ** 2
        
        R_desc_atom = np.row_stack((R_desc_val_atom,R_desc))
        R_d_desc_atom = R_d_desc_val_atom
        for j in range(n_val+n_train):
            
            blk_j = slice(j*dim_i , (j + 1)*dim_i )
            rj_desc_perms = np.reshape(
        np.tile(R_desc_atom[j, :], n_perms)[tril_perms_lin_mirror], (n_perms, -1), order='F'
    )
            
            if j<n_val:
                rj_d_desc = desc.d_desc_from_comp(R_d_desc_atom[j, :, :])[0][
                    :, keep_idxs_3n
                ]  # convert descriptor back to full representation
                # rj_d_desc 66 * 36
    
                rj_d_desc_perms = np.reshape(
                    np.tile(rj_d_desc.T, n_perms)[:, tril_perms_lin_mirror], (-1, dim_d, n_perms)
                )
                
                
            
            diff_ab_perms = np.empty((n_perms, dim_d))
            diff_ab_perms_t = np.empty((n_perms, dim_d))
            dim_i_keep =dim_i
            diff_ab_outer_perms = np.empty((dim_d, dim_i_keep)) 
            ri_d_desc = np.zeros((1, dim_d, dim_i))
            
            for i in range(n_val+n_train):
                blk_i = slice(i*dim_i , (i + 1)*dim_i )
                np.subtract(R_desc_atom[i, :], rj_desc_perms, out=diff_ab_perms)
                norm_ab_perms = sqrt5 * np.linalg.norm(diff_ab_perms, axis=1)
                
                if i<n_val:  
                    desc.d_desc_from_comp(R_d_desc_atom[i, :, :], out=ri_d_desc)
                    ri_desc_perms = np.reshape(
                        np.tile(R_desc_atom[i, :], n_perms)[tril_perms_lin_mirror], (n_perms, -1), order='F'
                    )
                    ri_d_desc_perms = np.reshape(
                np.tile(ri_d_desc[0].T, n_perms)[:, tril_perms_lin_mirror], (-1, dim_d, n_perms)
            )
                    np.subtract(R_desc_atom[j, :], ri_desc_perms, out=diff_ab_perms_t)
                    norm_ab_perms_t = sqrt5 * np.linalg.norm(diff_ab_perms_t, axis=1)
                    
                    K_fet = (
                        5
                        * diff_ab_perms_t
                        / (3 * sig ** 3)
                        * (norm_ab_perms_t[:, None] + sig)
                        * np.exp(-norm_ab_perms_t / sig)[:, None]
                    )
                    K_fet = np.einsum('ik,jki -> j', K_fet, ri_d_desc_perms)
                    K_FE_all[blk_i, j] = K_fet
                
                if j<n_val:
                    K_fe = (
                        5
                        * diff_ab_perms
                        / (3 * sig ** 3)
                        * (norm_ab_perms[:, None] + sig)
                        * np.exp(-norm_ab_perms / sig)[:, None]
                    )
                    K_fe = np.einsum('ik,jki -> j', K_fe, rj_d_desc_perms)
                    K_EF_all[i, blk_j] = K_fe[keep_idxs_3n]
                    
                    
                    
                    k = np.empty((dim_i, dim_i))
                    if i<n_val:
                        mat52_base_perms = np.exp(-norm_ab_perms / sig) / mat52_base_div * 5
                        
                        np.einsum(
                            'ki,kj->ij',
                            diff_ab_perms * mat52_base_perms[:, None] * 5,
                            np.einsum('ki,jik -> kj', diff_ab_perms, rj_d_desc_perms),
                            out=diff_ab_outer_perms
                        )
        
                        diff_ab_outer_perms -= np.einsum(
                            'ikj,j->ki',
                            rj_d_desc_perms,
                            (sig_pow2 + sig * norm_ab_perms) * mat52_base_perms,
                        )
                        
                        desc.d_desc_from_comp(R_d_desc_atom[i, :, :], out=ri_d_desc)
                        
                        np.dot(ri_d_desc[0].T, diff_ab_outer_perms, out=k)
                        K_FsFs_all[blk_i,blk_j]=-k
                

                K_EE_all[i,j]  =(
                1 + (norm_ab_perms / sig) * (1 + norm_ab_perms / (3 * sig))
            ).dot(np.exp(-norm_ab_perms / sig)) 
                
        
        n_batch=int(n_val/batch_size)
        
        for l in range(n_batch):
            R_E=np.empty((n_train+3*n_atoms*batch_size,n_train+3*n_atoms*batch_size))
            R_E[np.ix_(np.arange(n_train),np.arange(n_train))]=K_EE_all[
                np.ix_(np.arange(n_val,n_total),np.arange(n_val,n_total))].copy()
            
            Kr_E=np.empty((batch_size,n_train+3*n_atoms*batch_size))
            
            Y_vec=np.empty((n_train+3*n_atoms*batch_size))
            
            Y_vec[np.arange(n_train)]=-E_train.copy()
            index = np.arange(l*3*n_atoms*batch_size,(l+1)*3*n_atoms*batch_size)
            R_E[np.ix_(np.arange(n_train),np.arange(n_train,n_train+3*n_atoms*batch_size))]=K_EF_all[
                np.ix_(np.arange(n_val,n_total),index)].copy()
            R_E[np.ix_(np.arange(n_train,n_train+3*n_atoms*batch_size),np.arange(n_train))]=K_FE_all[
                np.ix_(index,np.arange(n_val,n_total))].copy()
            R_E[np.ix_(np.arange(n_train,n_train+3*n_atoms*batch_size),np.arange(n_train,n_train+3*n_atoms*batch_size))]=K_FsFs_all[
                np.ix_(index,index)].copy()
            Y_vec[np.arange(n_train,n_train+3*n_atoms*batch_size)]=F_star[index].copy()
            
            #index_E=np.concatenate((l,np.arange(n_val,n_total)))
            index_l=np.arange(l*batch_size,(l+1)*batch_size)
            Kr_E[np.ix_(np.arange(batch_size),np.arange(n_train))] = K_EE_all[np.ix_(index_l,np.arange(n_val,n_total))].copy()
            Kr_E[np.ix_(np.arange(batch_size),np.arange(n_train,n_train+3*n_atoms*batch_size))] =  K_EF_all[np.ix_(index_l,np.arange(l*3*n_atoms*batch_size,(l+1)*3*n_atoms*batch_size))].copy()
            
            add=np.diag(np.append(np.repeat(lam_e,n_train),np.repeat(lam_f,batch_size*n_atoms*3)))
            R_E +=add
            #R_E[np.diag_indices_from(R_E)] += lam_e
            
           
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    L, lower = sp.linalg.cho_factor(
                                    R_E, overwrite_a=True, check_finite=False
                                )
                    alpha_star=sp.linalg.cho_solve(
                        (L, lower), np.array(Y_vec), overwrite_b=True, check_finite=False
                    )
                except np.linalg.LinAlgError: 
                    alpha_star  = sp.linalg.solve(
                            R_E,  Y_vec, overwrite_a=True, overwrite_b=True, check_finite=False
                        )
            E_predict[index_l]=np.matmul(Kr_E,alpha_star)

        return E_predict
    
   

dataset=np.load('benzene_old_dft.npz')
#dataset=np.load('uracil_dft.npz')
#dataset=np.load('malonaldehyde_ccsd_t-train.npz')
#dataset=np.load('glucose_alpha.npz')
#dataset=np.load('alkane.npz')
#dataset=np.load('aspirin_dft.npz')
#dataset=np.load('naphthalene_dft.npz')
#dataset=np.load('aspirin_ccsd-train.npz')
#
gdml_train=GDMLTrain()
# #n_train=np.array([100])
n_train=np.array([100])
for i in range(n_train.shape[0]):
    print(' The N_train is '+repr(n_train[i])+'--------------------')
    task=gdml_train.create_task(dataset,n_train[i],dataset,10,10,100,1e-15,use_E_cstr=True,batch_size=20)
    k1,k2 = gdml_train.train(task)
    #test_MAE=gdml_train.test(task,sig_opt,sig_opt_E,alphas_opt)

# n_train=100
# # use_sym=False means use GDML, use_sym=True (default ) means use sGDML
# task=gdml_train.create_task(dataset,n_train,dataset,50,50,100,1e-15,use_E_cstr=True,batch_size=10)
# # #task=gdml_train.create_task(dataset,100,dataset,50,100,100,1e-15)

# sig_opt,sig_opt_E,alphas_opt = gdml_train.train(task)

# test_MAE=gdml_train.test(task,sig_opt,sig_opt_E,alphas_opt)

import seaborn as sns
#import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# #index_x=np.arange(36*)

# correlations =k1 #计算变量之间的相关系数矩阵
# # plot correlation matrix
# fig = plt.figure() #调用figure创建一个绘图对象
# ax = fig.add_subplot(111)
# cax = ax.matshow(correlations, cmap='gist_gray',vmin=np.min(k1), vmax=np.max(k1))  #绘制热力图，从-1到1
# fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
# #ticks = np.arange(0,9,1) #生成0-9，步长为1
# #ax.set_xticks(ticks)  #生成刻度
# #ax.set_yticks(ticks)
# #ax.set_xticklabels(names) #生成x轴标签
# #ax.set_yticklabels(names)
# #plt.imshow(I, cmap='gray');
# plt.show()
n_train=2
correlations =k1[np.ix_(np.arange(36*n_train),np.arange(36*n_train))] #计算变量之间的相关系数矩阵
# plot correlation matrix
fig = plt.figure() #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, cmap='gist_gray',vmin=np.min(k1), vmax=np.max(k1))  #绘制热力图，从-1到1
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
plt.show()

index_diff_atom=task['index_diff_atom']
ind_i=0
index_eg=np.tile(np.arange(3),len(index_diff_atom[ind_i]))+3*np.repeat(index_diff_atom[ind_i],3)

n_train=5
                
#index_x=np.repeat(np.arange(n_val)*(dim_i),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_val)
index_y=np.repeat(np.arange(n_train)*(36),3*len(index_diff_atom[ind_i]))+np.tile(index_eg,n_train)

    # index_x=np.repeat(np.arange(n_val)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_val)
    # index_y=np.repeat(np.arange(n_train)*dim_i,3)+np.tile(np.array([3*ind_i,3*ind_i+1,3*ind_i+2]),n_train)

k2=k1[np.ix_(index_y,index_y)]
correlations =k2 #计算变量之间的相关系数矩阵
# plot correlation matrix
fig = plt.figure() #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, cmap='gist_gray',vmin=-0.012, vmax=0.03)  #绘制热力图，从-1到1
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
#ticks = np.arange(0,9,1) #生成0-9，步长为1
#ax.set_xticks(ticks)  #生成刻度
#ax.set_yticks(ticks)
#ax.set_xticklabels(names) #生成x轴标签
#ax.set_yticklabels(names)
#plt.imshow(I, cmap='gray');
plt.show()

# a = k1
# fig, ax = plt.subplots(figsize = (9,9))
# #二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
# #和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
# sns.heatmap(pd.DataFrame(np.round(a,2), columns = ['a', 'b', 'c'], index = range(1,5)), 
#                 annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
# #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
# #            square=True, cmap="YlGnBu")
# ax.set_title('二维数组热力图', fontsize = 18)
# ax.set_ylabel('数字', fontsize = 18)
# ax.set_xlabel('字母', fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
import pandas as pd
import matplotlib.pyplot as plt

n_train=np.array([200,400,600,800,1000,1200,1400])
MAE_F_AFF=np.array([0.9315,	0.5533,	0.4255	,0.3496,	0.3138,	0.2827,	0.255])
MAE_E_AFF=np.array([0.213,	0.1488,	0.1359,	0.1238,	0.126,	0.1004	,0.088])
Time_AFF=np.array([0.4825,	1.15,	2.32,	4.319,	5.94,	8.58,	12.36])

MAE_F_sGDML=np.array([0.6688,	0.3947,	0.3157,	0.265,	0.2468])
MAE_E_sGDML=np.array([0.1413,	0.1141,	0.109,	0.11,	0.1])
Time_sGDML=np.array([1.06,	4.74,	12.69,	26.87,	39])

avocado=pd.DataFrame(np.array([n_train,Time_AFF]),index=['n_train','training time'])
fig, ax = plt.subplots(figsize = (6,4))
avocado.plot(kind = "bar")
plt.show()

#fig = plt.figure()
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
width=100
#cmap = plt.get_cmap('cubehelix')
#ax2 = fig.add_axes([0,0,1,1])
#ax1.plot( MAE_F_sGDML, color='peachpuff')
ax1.plot(n_train[np.arange(5)], MAE_F_sGDML, color='salmon')
ax1.plot(n_train, MAE_F_AFF, color='darkblue')
ax2.bar(n_train[np.arange(5)], Time_sGDML, width, color='salmon',alpha=0.4)
ax2.bar(n_train, Time_AFF, width, color='darkblue',alpha=0.4)
ax2.set_ylabel('Time: seconds')
#ax2.set_title('number of training samples')
#ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
ax2.set_yticks(np.arange(0, 40, 5))
ax2.legend(labels=['sGDML','AFF'])
fig.suptitle('The MAE of Force and tthe training time ', fontsize=16)
plt.show()

width = 100 # width of a bar
m1_t = pd.DataFrame({
 'n_train' : n_train,
 'MAE_F_AFF' : MAE_F_AFF,
 'MAE_E_AFF' : MAE_E_AFF,
 'Time_AFF' : Time_AFF,
 'MAE_F_sGDML' : MAE_F_sGDML,
 'MAE_E_sGDML' : MAE_E_sGDML,
 'Time_sGDML' : Time_sGDML})

m1_t[['Time_AFF','Time_sGDML']].plot(kind='bar', width = width)
#m1_t['MAE_F_AFF'].plot(secondary_y=True)
#m1_t['MAE_F_AFF','Time_sGDML'].plot(secondary_y=True)

ax = plt.gca()
plt.xlim([0, 1400])
ax.set_xticklabels(n_train)

plt.show()








