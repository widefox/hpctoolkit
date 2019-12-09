// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2017, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

#ifndef __DATACENTRIC_DATA_TREE_H__
#define __DATACENTRIC_DATA_TREE_H__

#include <cct/cct.h>


/******************************************************************************
 * macros
 *****************************************************************************/

#define DATA_STATIC_MAGIC 0xFEA12B0B
#define DATA_DYNAMIC_MAGIC 0x68706374



/******************************************************************************
 * type definitions
 *****************************************************************************/

typedef enum datatree_info_status_e {
  DATATREE_INFO_UNHANDLED,
  DATATREE_INFO_HANDLED
} datatree_info_status_t;

typedef struct datatree_info_s {
  long        magic;
  cct_node_t *context;
  size_t      bytes;
  void       *memblock;
  void       *rmemblock;	// additional information to record remote memory

  datatree_info_status_t status;

  struct datatree_info_s *left;
  struct datatree_info_s *right;
} datatree_info_t;

/* generic insert info item into a data tree*/
struct datatree_info_s*
datatree_info_insert_ext(struct datatree_info_s **data_root,
                         spinlock_t *data_lock,
                         struct datatree_info_s *node);

/* generic lookup data tree */
struct datatree_info_s *
datatree_info_lookup_ext( struct datatree_info_s **data_root,
                      spinlock_t *lock,
                      void *key, void **start, void **end);

/* * Insert a node */ 
void datatree_info_insert(struct datatree_info_s *node);

/* find a cct node for a given key and range */
struct datatree_info_s * datatree_info_lookup(void *key, void **start, void **end);

/* remove a node containing a memory block */
struct datatree_info_s * datatree_info_delete(void *memblock);

#endif //__DATACENTRIC_DATA_TREE_H__