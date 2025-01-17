/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Nov 14, 2017
 *      Author: Sharmin Rahman

 *********************************************************************************/

/**
 * @file DepthFrameSynchronizer.cpp
 * @brief Source file for the DepthFrameSynchronizer class.
 * @author Sharmin Rahman
 */

#include "okvis/DepthFrameSynchronizer.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {

DepthFrameSynchronizer::DepthFrameSynchronizer()
  : shutdown_(false) {}

DepthFrameSynchronizer::~DepthFrameSynchronizer() {
  if(!shutdown_)
    shutdown();
}

// Tell the synchronizer that a new Depth measurement has been registered.
void DepthFrameSynchronizer::gotDepthData(const okvis::Time& stamp) {
  newestDepthDataStamp_ = stamp;
  if(depthDataNeededUntil_ < stamp)
    gotNeededDepthData_.notify_all();
}

// Wait until a Depth measurement with a timestamp equal or newer to the supplied one is registered.
bool DepthFrameSynchronizer::waitForUpToDateDepthData(const okvis::Time& frame_stamp) {
  // if the newest depth data timestamp is smaller than frame_stamp, wait until
  // depth_data newer than frame_stamp arrives
  if(newestDepthDataStamp_ <= frame_stamp && !shutdown_) {
    depthDataNeededUntil_ = frame_stamp;
    std::unique_lock<std::mutex> lock(mutex_);
    gotNeededDepthData_.wait(lock);
  }
  if(shutdown_)
    return false;
  return true;
}

// Tell the synchronizer to shutdown. This will notify all waiting threads to wake up.
void DepthFrameSynchronizer::shutdown() {
  shutdown_ = true;
  gotNeededDepthData_.notify_all();
}

} /* namespace okvis */
