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

//
// SYNC sample source simple oo interface
//

/******************************************************************************
 * system includes
 *****************************************************************************/

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>



/******************************************************************************
 * libmonitor
 *****************************************************************************/

#include <monitor.h>



/******************************************************************************
 * local includes
 *****************************************************************************/

#include <hpcrun/hpcrun_options.h>
#include <hpcrun/disabled.h>
#include <hpcrun/metrics.h>
#include <sample_event.h>
#include "sample_source_obj.h"
#include "common.h"
#include <hpcrun/sample_sources_registered.h>
#include "simple_oo.h"
#include <hpcrun/thread_data.h>
#include <utilities/tokenize.h>

#include <messages/messages.h>



/******************************************************************************
 * method definitions
 *****************************************************************************/

static void
METHOD_FN(init)
{
  self->state = INIT; // no actual init actions necessary for sync events
}

static void
METHOD_FN(thread_init)
{
  TMSG(SYNC_CTL, "thread init (no action needed)");
}

static void
METHOD_FN(thread_init_action)
{
  TMSG(SYNC_CTL, "thread action (noop)");
}


static void
METHOD_FN(start)
{
  TMSG(SYNC_CTL,"starting SYNC");

  TD_GET(ss_state)[self->sel_idx] = START;
}

static void
METHOD_FN(thread_fini_action)
{
  TMSG(SYNC_CTL, "thread fini (no action needed");
}

static void
METHOD_FN(stop)
{
  TMSG(SYNC_CTL,"stopping SYNC");
  TD_GET(ss_state)[self->sel_idx] = STOP;
}


static void
METHOD_FN(shutdown)
{
  METHOD_CALL(self,stop); // make sure stop has been called
  self->state = UNINIT;
}


static bool
METHOD_FN(supports_event,const char *ev_str)
{
  return hpcrun_ev_is(ev_str,"SYNC");
}
 

//
// Special SYNC protocol:
//  if event is SYNC@xxx, then create xxx events
//  if event is just plain SYNC, then default to 1 event
//
static void
METHOD_FN(process_event_list,int lush_metrics)
{
  char *_p = strchr(METHOD_CALL(self,get_event_str), '@');
  int n_events = 1;
  if (_p) {
    n_events = atoi(_p+1);
  }
#ifdef OLD_DEFAULT
  if (! n_events ) {
    n_events = 1;
  }
#endif // OLD_DEFAULT
  kind_info_t *sync_kind = hpcrun_metrics_new_kind();
  for (int i = 0; i < n_events; i++)
    hpcrun_set_new_metric_info(sync_kind, "RENAME");
  hpcrun_close_kind(sync_kind);
}


//
// Event sets not relevant for this sample source
// Events are generated by user code
//
static void
METHOD_FN(gen_event_set,int lush_metrics)
{
}


static void
METHOD_FN(display_events)
{
  printf("===========================================================================\n");
  printf("Available synchronous events\n");
  printf("===========================================================================\n");
  printf("Name\t\tDescription\n");
  printf("---------------------------------------------------------------------------\n");
  printf("SYNC\t\tThe number of synchronous metric slots allocated,\n"
	 "\t\teg, SYNC@3 would generate 3 slots\n");
  printf("\n");
}

/***************************************************************************
 * object
 ***************************************************************************/

//
// sync class is "SS_SOFTWARE" so that both synchronous and asynchronous sampling is possible
// 

#define ss_name sync
#define ss_cls SS_SOFTWARE

#include "ss_obj.h"
