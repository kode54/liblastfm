/*
   Copyright 2005-2009 Last.fm Ltd. <mir@last.fm>

   This file is part of liblastfm.

   liblastfm is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   liblastfm is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with liblastfm.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "OptFFT.h"
#include "fp_helper_fun.h"
#include "Filter.h" // for NBANDS

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <cstdlib> 
#include <stdexcept>
#include <cstring>

using namespace std;
// ----------------------------------------------------------------------

namespace fingerprint
{

// -----------------------------------------------------------------------------

static void *_memalign_malloc(size_t size, size_t align) {
   void *ret = NULL;
   if(posix_memalign(&ret, align, size) != 0) {
      return NULL;
   }
   return ret;
}

static int _malloc_complex(DSPSplitComplex *out, size_t count) {
   out->realp = static_cast<float *>( _memalign_malloc( sizeof(float) * count, 16 ) );
   out->imagp = static_cast<float *>( _memalign_malloc( sizeof(float) * count, 16 ) );
   if( out->realp && out->imagp ) return 0;
   else return -1;
}

static void _free_complex(DSPSplitComplex *in) {
   if(in->realp) free(in->realp);
   if(in->imagp) free(in->imagp);
   in->realp = NULL;
   in->imagp = NULL;
}

// -----------------------------------------------------------------------------

OptFFT::OptFFT(const size_t maxDataSize)
{
   assert( maxDataSize % OVERLAPSAMPLES == 0 );

   // DOUBLE
   //m_pIn = static_cast<double*>( fftw_malloc(sizeof(double) * FRAMESIZE) );
   //m_pOut = static_cast<fftw_complex*>( fftw_malloc(sizeof(fftw_complex) * (FRAMESIZE/2 + 1)) );
    //m_p = fftw_plan_dft_r2c_1f(FRAMESIZE, m_pIn, m_pOut, FFTW_ESTIMATE); // FFTW_ESTIMATE or FFTW_MEASURE

   // FLOAT
 //  m_pIn = static_cast<float*>( fftwf_malloc(sizeof(float) * FRAMESIZE) );
 //  m_pOut = static_cast<fftwf_complex*>( fftwf_malloc(sizeof(fftwf_complex) * (FRAMESIZE/2 + 1)) );

    //// in destroyed when line executed
    //m_p = fftwf_plan_dft_r2c_1d(FRAMESIZE, m_pIn, m_pOut, FFTW_ESTIMATE); // FFTW_ESTIMATE or FFTW_MEASURE

   m_hann = static_cast<float *> ( _memalign_malloc(sizeof(float) * FRAMESIZE, 16 ) );
   if ( !m_hann )
   {
      ostringstream oss;
      oss << "_memalign_malloc failed on m_hann. Trying to allocate <"
          << sizeof(float) * FRAMESIZE
          << "> bytes";
      throw std::runtime_error(oss.str());
   }

   vDSP_hann_window( m_hann, FRAMESIZE, vDSP_HANN_DENORM );

   //-----------------------------------------------------------------

   int numSamplesPerFrame    = FRAMESIZE;
   int numSamplesPerFrameOut = FRAMESIZE/2+1;

    m_maxFrames = static_cast<int> ( (maxDataSize - FRAMESIZE) / OVERLAPSAMPLES + 1 );

   m_pIn  = static_cast<float*> ( _memalign_malloc(sizeof(float) * (numSamplesPerFrame * m_maxFrames), 16 ) );
   if ( !m_pIn )
   {
      ostringstream oss;
      oss << "_memalign_malloc failed on m_pIn. Trying to allocate <"
          << sizeof(float) * (numSamplesPerFrame * m_maxFrames)
          << "> bytes";
      throw std::runtime_error(oss.str());
   }

   m_pOut = static_cast<DSPSplitComplex*>( _memalign_malloc(sizeof(DSPSplitComplex) * m_maxFrames, 16 ) );
   if ( !m_pOut )
   {
      ostringstream oss;
      oss << "_memalign_malloc failed on m_pOut. Trying to allocate <"
          << sizeof(DSPSplitComplex) * m_maxFrames
          << "> bytes";

      throw std::runtime_error(oss.str());
   }

   bzero( m_pOut, sizeof(DSPSplitComplex) * m_maxFrames );

   for(size_t i = 0; i < m_maxFrames; ++i)
   {
      int retcode = _malloc_complex( &m_pOut[i], numSamplesPerFrameOut );
      if( retcode != 0 )
      {
         ostringstream oss;
         oss << "_malloc_complex failed on m_pOut["
             << i
             << "]. Trying to allocate 2x <"
             << sizeof(float) * numSamplesPerFrameOut
             << "> bytes";
         throw std::runtime_error(oss.str());
      }
   }

   m_p = vDSP_DFT_zrop_CreateSetup(nil, numSamplesPerFrame, vDSP_DFT_FORWARD);

   if ( !m_p )
      throw std::runtime_error ("vDSP_DFT_zrop_CreateSetup failed");

   double base = exp( log( static_cast<double>(MAXFREQ) / static_cast<double>(MINFREQ) ) / 
                      static_cast<double>(Filter::NBANDS) 
                    );

   m_powTable.resize( Filter::NBANDS+1 );
   for ( unsigned int i = 0; i < Filter::NBANDS + 1; ++i )
      m_powTable[i] = static_cast<unsigned int>( (pow(base, static_cast<double>(i)) - 1.0) * MINCOEF );

   m_pFrames = new float*[m_maxFrames];

   if ( !m_pFrames )
   {
      ostringstream oss;
      oss << "Allocation failed on m_pFrames. Trying to allocate <" 
         << sizeof(float*) * m_maxFrames
         << "> bytes";

      throw std::runtime_error(oss.str());
   }

   for (int i = 0; i < m_maxFrames; ++i) 
   {
      m_pFrames[i] = new float[Filter::NBANDS];
      if ( !m_pFrames[i] )
         throw std::runtime_error("Allocation failed on m_pFrames");
   }

}

// ----------------------------------------------------------------------

OptFFT::~OptFFT()
{
   if(m_p) vDSP_DFT_DestroySetup(m_p);

   if(m_pIn) free(m_pIn);

   if(m_pOut)
   {
      for (size_t i = 0; i < m_maxFrames; ++i)
      {
         _free_complex(&m_pOut[i]);
      }
      free(m_pOut);
   }

   for (int i = 0; i < m_maxFrames; ++i)
      delete [] m_pFrames[i];

   delete [] m_pFrames;

   if(m_hann) free(m_hann);
}

// ----------------------------------------------------------------------

int OptFFT::process(float* pInData, const size_t dataSize)
{
   // generally is the same of the one we used in the constructor (m_maxFrames) but
   // might be less at the end of the stream
   int nFrames = static_cast<int>( (dataSize - FRAMESIZE) / OVERLAPSAMPLES + 1 );

   float* pIn_It = m_pIn;

   for (int i = 0; i < nFrames; ++i)
   {
      memcpy( pIn_It, &pInData[i*OVERLAPSAMPLES], sizeof(float) * FRAMESIZE);
      // apply hanning window
      applyHann(pIn_It, FRAMESIZE);

      pIn_It += FRAMESIZE;
   }

   // fill the rest with zeroes
   if ( nFrames < m_maxFrames )
      memset( pIn_It, 0, sizeof(float) * (m_maxFrames-nFrames) * FRAMESIZE );

   float scalingFactor = static_cast<float>(FRAMESIZE) / 2.0f;

   for (size_t i = 0; i < m_maxFrames; ++i)
   {
      vDSP_ctoz((DSPComplex *)(pIn_It + i * FRAMESIZE), 2, &m_pOut[i], 1, FRAMESIZE / 2);
      vDSP_DFT_Execute(m_p, m_pOut[i].realp, m_pOut[i].imagp, m_pOut[i].realp, m_pOut[i].imagp);

      // scaling (?)
      vDSP_vsdiv(m_pOut[i].realp, 1, &scalingFactor, m_pOut[i].realp, 1, FRAMESIZE);
      vDSP_vsdiv(m_pOut[i].imagp, 1, &scalingFactor, m_pOut[i].imagp, 1, FRAMESIZE);
   }

   int frameStart;
   unsigned int outBlocStart;
   unsigned int outBlocEnd;

    for (int i = 0; i < nFrames; ++i) 
   {
       // compute bands
       for (unsigned int j = 0; j < Filter::NBANDS; j++) 
      {
         outBlocStart = m_powTable[j];
         outBlocEnd   = m_powTable[j+1];

           m_pFrames[i][j] = 0;

           // WARNING: We're double counting the last one here.
           // this bug is to match matlab's implementation bug in power2band.m
         unsigned int end_k = outBlocEnd + static_cast<unsigned int>(MINCOEF);
           for (unsigned int k = outBlocStart + static_cast<unsigned int>(MINCOEF); k <= end_k; k++) 
         {
               m_pFrames[i][j] += m_pOut[i].realp[k] * m_pOut[i].realp[k] +
                               m_pOut[i].imagp[k] * m_pOut[i].imagp[k];
           }

           // WARNING: if we change the k<=end to k<end above, we need to change the following line
           m_pFrames[i][j] /= static_cast<float>(outBlocEnd - outBlocStart + 1);        
       }   
   }

   return nFrames;
}

// -----------------------------------------------------------------------------

void OptFFT::applyHann( float* pInData, const size_t dataSize )
{
   assert (dataSize == 2048);

   vDSP_vmul( pInData, 1, m_hann, 1, pInData, 1, dataSize );
}

// -----------------------------------------------------------------------------

} // end of namespace

// ----------------------------------------------------------------------
