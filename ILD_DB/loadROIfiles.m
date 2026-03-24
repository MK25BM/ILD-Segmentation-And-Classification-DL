function ROIs = loadROIfiles(ROIfileName)
% Loads all the ROIs stored in the *.txt file from the specified location.
% The ROIs are returned as a structure array as:
% ROIs(i).spacing_x (double)
% ROIs(i).spacing_y (double)
% ROIs(i).label (string)
% ROIs(i).slice_number (double)
% ROIs(i).xValues (vector of double)
% ROIs(i).yValues (vector of double)
%
% USAGE :
% ROIs = loadROIfiles(ROIfileName)
%
%
% @author University & University Hospitals of Geneva (HUG) MedGIFT Group
%         Adrien Depeursinge adrien.depeursinge@hevs.ch
%
% 	   Copyright (C) 2012, MedGIFT group at University Hospitals of Geneva
%       
%      This program is free software; you can redistribute it and/or modify
%	   it under the terms of the GNU General Public License as published by
%      the Free Software Foundation; either version 2 of the License, or
%      (at your option) any later version.
%
%      This program is distributed in the hope that it will be useful,
%      but WITHOUT ANY WARRANTY; without even the implied warranty of
%      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%      GNU General Public License for more details.
%
%      You should have received a copy of the GNU General Public License
%      along with this program; if not, write to the Free Software
%      Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA 
%

fid = fopen(ROIfileName, 'r');
i=1;
while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    
    if (isempty(findstr(tline, 'SpacingX:')) == false)
        spacing_x = str2double(strrep(tline, 'SpacingX: ', ''));
    elseif (isempty(findstr(tline, 'SpacingY:')) == false)
        spacing_y = str2double(strrep(tline, 'SpacingY: ', ''));
    elseif (isempty(findstr(tline, 'label:')) == false)
        label = strrep(tline, 'label: ', '');
    elseif (isempty(findstr(tline, 'slice_number:')) == false)
        slice_number = str2double(strrep(tline, 'slice_number: ', ''));
    elseif (isempty(findstr(tline, 'nb_points_on_contour:')) == false)
        nb_points_on_contour = str2double(strrep(tline, 'nb_points_on_contour: ', ''));
        xValues = [];
        yValues = [];
        for j=1:nb_points_on_contour
            tline = fgetl(fid);
            [xValStr, yValStr] = strtok(tline, ' ');
            xVal = str2double(xValStr);
            yVal = str2double(yValStr);
            xValues = [xValues xVal];
            yValues = [yValues yVal];
        end
        % store the ROI into the struct array "ROIs"
        ROIs(i).spacing_x = spacing_x;
        ROIs(i).spacing_y = spacing_y;
        ROIs(i).label = label;
        ROIs(i).slice_number = slice_number;
        ROIs(i).xValues = xValues;
        ROIs(i).yValues = yValues;

        i=i+1;
    end
end

fclose(fid);