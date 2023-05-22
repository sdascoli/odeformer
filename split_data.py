# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import os
import sys
import math


if __name__ == '__main__':

    assert len(sys.argv) == 3

    data_path = sys.argv[1]
    tst_path = sys.argv[1] + '.test'
    tst_size = int(sys.argv[2])
    #assert not os.path.isfile(tst_path)
    assert tst_size > 0

    print(f"Reading data from {data_path} ...")
    with io.open(data_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    print(f"Read {len(lines)}")
    with io.open(data_path, mode='w', encoding='utf-8') as f:
        f.writelines(lines[tst_size:])

    print(f"Writing test data to {tst_path} ...")
    f_test = io.open(tst_path, mode='w', encoding='utf-8')

    for i, line in enumerate(lines[:tst_size]):
        f_test.write(line)

    f_test.close()