#!/usr/bin/env python


#
#   Copyright (c) 2011
#     Patrick Sudowe	<sudowe@umic.rwth-aachen.de>
#     RWTH Aachen University, Germany
#
#   This file is part of groundHOG.
#
#   GroundHOG is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GroundHOG is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with groundHOG.  If not, see <http://www.gnu.org/licenses/>.
#

import os, sys, glob
import Image, ImageDraw

__usage__= """visualize_multidetections.py <DETECTIONS_DIR> <PATH_TO_IMG> <OUTPUT_DIR>"""

if len(sys.argv) <> 4:
	print __usage__
	sys.exit(1)

DETECTIONS_DIR = sys.argv[1]
PATH = sys.argv[2]
OUT_DIR = sys.argv[3]

WIDTH = 2

def draw_thick_rectangle(draw, rect, color):
	x,y,mx,my = rect
	draw.line((x,y,mx,y), width=WIDTH, fill=color)
	draw.line((x,my,mx,my), width=WIDTH, fill=color)
	draw.line((x,y,x,my), width=WIDTH, fill=color)
	draw.line((mx,y,mx,my), width=WIDTH, fill=color)



# parse multi detections file
# returns a list of list of detection tuples
def parse_multi_detections(filename):
	f = open(filename)
	multi_detections = {}
	current_model_name = ''
	for line in f.readlines():
		if len(line.strip()) == 0:
			continue
		if line.find('#') >= 0:
			if line.find('model') >= 0:
				current_model_name = line.split()[2]
				multi_detections[current_model_name] = []
		else:
			el = line.split()
			if len(el) == 5: # regular line with a detection
				x = int(el[0])
				y = int(el[1])
				ux = int(el[2])
				uy = int(el[3])
				score = float(el[4])
				print '\t', x,y,ux,uy, '\t', score
				multi_detections[current_model_name].append((x,y,ux,uy,score))
	return multi_detections


def visualize_multi_detections(multi_detections, fnImage, fnOut):
	img = Image.open(fnImage)
	draw = ImageDraw.Draw(img)

	for model in multi_detections.keys():
		for d in multi_detections[model]:
#			label = (int(model) -1) *30
			label = model
			color = (255,0,0)
			if label == "1":
				color = (255,0,0)
			elif label == "2":
				color = (0,255,0)
			elif label == "3":
				color = (0,0,255)
			elif label == "4":
				color = (255,255,0)
			elif label == "5":
				color = (0,255,255)
			elif label == "6":
				color = (255,0,255)
			elif label == "7":
				color = (0,0,0)
			else:
				color = (255,255,255)

#			draw.rectangle([(d[0],d[1]),(d[2],d[3])], outline=(255,0,0))
			draw_thick_rectangle(draw, (d[0],d[1],d[2],d[3]), color)
			draw.text([(d[0]+2,d[3]-16)], str(label), fill=color)

	del draw
	img.save(fnOut)


for file in glob.glob( os.path.join( DETECTIONS_DIR, '*.detections' ) ):
	print "\n--------------------------------------------------"
	elems = os.path.basename(file).split(os.path.extsep)
	fnImage = elems[0] + '.' + elems[1]
	print "image file:", fnImage

	fnOut = os.path.join( OUT_DIR, fnImage )
	fnImage = os.path.join( PATH, fnImage )

	detections = parse_multi_detections( file )

	visualize_multi_detections( detections, fnImage, fnOut)

