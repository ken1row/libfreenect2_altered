#!/usr/bin/env python

import shutil
import os
import glob

#t = glob.glob('*.py')
#t.extend(glob.glob('*.cmake'))
#print t

def backup(dirname):
    print 'backup target:', dirname
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    mv_filenames = glob.glob('*.png')
    cp_filenames = glob.glob('*.py')
    cp_filenames.extend(glob.glob('*.dat*'))
    mv_filenames.extend(glob.glob('*.npy'))
    exception_files = ['valid_pixels.png']
    for fn in mv_filenames:
        dst = os.path.join(dirname, os.path.basename(fn))
        # Exception
        if fn in exception_files:
            shutil.copy(fn, dst)
        else:
            shutil.move(fn, dst)
    for fn in cp_filenames:
        dst =  os.path.join(dirname, os.path.basename(fn))
        shutil.copy(fn, dst)
    
def clear():
    print 'Clear current dir.'
    cp_filenames = glob.glob('*.py')
    cp_filenames.extend(glob.glob('*.png'))
    cp_filenames.extend(glob.glob('*.dat*'))
    cp_filenames.extend(glob.glob('*.npy'))
    exception_files = ['valid_pixels.png']
    for fn in cp_filenames:
        if fn in exception_files:
            continue
        os.remove(fn)
    

def new_backup(d='data', prefix='test', zero_pad=2):
    i = 0
    while True:
        dirname = os.path.join(d, prefix+str(i).zfill(zero_pad))
        if os.path.exists(dirname):
            i += 1
            continue
        backup(dirname)
        break
    
def roll_back(d='data', targ=''):
    dirname = os.path.join(d, targ)
    if not os.path.exists(dirname):
        print 'Target directory does not exist.', dirname
        return
    filenames = glob.glob(os.path.join(dirname, '*.png'))
    filenames.extend(glob.glob(os.path.join(dirname, '*.dat')))
    exception_files = ['valid_pixels.png']
    for fn in filenames:
        dst = os.path.basename(fn)
        if dst in exception_files:
            continue
        shutil.copy(fn, dst)
    f = open('backuplog.rollback', 'w')
    f.write(dirname)
    f.close()
    print 'rollback:', dirname
        
def backup_again():
    if not os.path.exists('backuplog.rollback'):
        print 'Rollback metadata does not exists.'
        return
    f = open('backuplog.rollback', 'r')
    dirname = f.read()
    f.close()
    backup(dirname)
    os.remove('backuplog.rollback')
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backup and checkout experiments.')
    parser.add_argument('-r', '--rollback', default='', help='Rollback former experiment data.')
    parser.add_argument('-c', '--clear', action='store_true', help='Clear current directory.')
    parser.add_argument('-d', '--dir', default='data', help='Data directory path.')
    parser.add_argument('-p', '--prefix', default='test', help='Directory name (prefix).')
    parser.add_argument('-z', '--zeropad', type=int, default=2, help='Digit length of directory numbering.')
    args = parser.parse_args()
    
    if len(args.rollback) > 0:
        roll_back(d=args.dir, targ=args.rollback)
    elif args.clear:
        clear()
    elif os.path.exists('backuplog.rollback'):
        backup_again()
    else:
        new_backup(d=args.dir, prefix=args.prefix, zero_pad=args.zeropad)