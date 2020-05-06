# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
r"""Generate samples with a pretrained GANSynth model.

To use a config of hyperparameters and manual hparams:
#>>> python magenta/models/gansynth/generate.py \
#>>> --ckpt_dir=/path/to/ckpt/dir --output_dir=/path/to/output/dir \
#>>> --midi_file=/path/to/file.mid

If a MIDI file is specified, notes are synthesized with interpolation between
latent vectors in time. If no MIDI file is given, a random batch of notes is
synthesized.
"""

import os

import absl.flags
from magenta.models.gansynth.lib import flags as lib_flags
from magenta.models.gansynth.lib import generate_util as gu
from magenta.models.gansynth.lib import model as lib_model
from magenta.models.gansynth.lib import util
import tensorflow.compat.v1 as tf

import numpy as np
import time
import deepbeats.gen_z as gen_z

absl.flags.DEFINE_string('ckpt_dir',
                         '/tmp/gansynth/acoustic_only',
                         'Path to the base directory of pretrained checkpoints.'
                         'The base directory should contain many '
                         '"stage_000*" subdirectories.')
absl.flags.DEFINE_string('output_dir',
                         '/tmp/gansynth/samples',
                         'Path to directory to save wave files.')
absl.flags.DEFINE_string('midi_file',
                         '',
                         'Path to a MIDI file (.mid) to synthesize.')
absl.flags.DEFINE_integer('batch_size', 8, 'Batch size for generation.')
absl.flags.DEFINE_float('secs_per_instrument', 6.0,
                        'In random interpolations, the seconds it takes to '
                        'interpolate from one instrument to another.')

FLAGS = absl.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)

def make_output(model, model_name, midi_name='nomidiname', gen_z_mode='', seed=0):
  time_prefix = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
  np.random.seed(seed)

  # Make an output directory if it doesn't exist
  output_dir = util.expand_path(FLAGS.output_dir)
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  if FLAGS.midi_file:
    # If a MIDI file is provided, synthesize interpolations across the clip
    unused_ns, notes = gu.load_midi(FLAGS.midi_file)

    # Could be moved out of the if for no midi case
    name_prefix = '_'.join([time_prefix, model_name, midi_name, gen_z_mode, str(seed)])
    genz = gen_z.Gen_Z()
    genz.mode = gen_z_mode

    # Distribute latent vectors linearly in time
    z_instruments, t_instruments = gu.get_random_instruments(
        model,
        notes['end_times'][-1], genz,
        secs_per_instrument=FLAGS.secs_per_instrument)

    genz.write_matrix_to_file(z_instruments, output_dir, name_prefix)

    # Get latent vectors for each note
    z_notes = gu.get_z_notes(notes['start_times'], z_instruments, t_instruments)

    # Generate audio for each note
    print('Generating {} samples...'.format(len(z_notes)))
    audio_notes = model.generate_samples_from_z(z_notes, notes['pitches'])

    # Make a single audio clip
    audio_clip = gu.combine_notes(audio_notes,
                                  notes['start_times'],
                                  notes['end_times'],
                                  notes['velocities'])

    # Write the wave files
    fname = os.path.join(output_dir, name_prefix + '.wav')
    gu.save_wav(audio_clip, fname)
  else:
    # Otherwise, just generate a batch of random sounds
    waves = model.generate_samples(FLAGS.batch_size)
    # Write the wave files
    name_prefix = '_'.join([time_prefix, model_name, str(seed)])
    for i in range(len(waves)):
      fname = os.path.join(output_dir, name_prefix + 'nomidifile_{}.wav'.format(i))
      gu.save_wav(waves[i], fname)

def main(unused_argv):

  model_name = '9instr'
  midi_name = 'Bourree_7octaves'
  gen_z_modes = ['arithmetic_radius', 'square_radius', 'exp_radius']
  seeds = range(3)

  absl.flags.FLAGS.alsologtostderr = True

  # Load the model
  flags = lib_flags.Flags({'batch_size_schedule': [FLAGS.batch_size]})
  model = lib_model.Model.load_from_path(FLAGS.ckpt_dir, flags)

  for gen_z_mode in gen_z_modes:
    for seed in seeds:
      make_output(model, model_name, midi_name, gen_z_mode, seed)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
