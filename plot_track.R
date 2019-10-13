path_root = '~/Documents/FOD/DMRI_code/Results_new/ROI_HCPM2'
setwd(path_root)
library(RcppCNPy)
library(rgl)
source('~/Documents/FOD/FOD_code/dwi_track.R')

name = 'retest_123K_3e3_b4_sm6'
tmp = unzip(paste0('track_',name,'.npz'))
temp = list()
for(i in 1:length(tmp)) {
  temp[[substr(tmp[i], 3, nchar(tmp[i])-4)]] = npyLoad(tmp[i])
}
temp$vec = matrix(temp$vec, length(temp$vec)/3, 3)
temp$loc = matrix(temp$loc, length(temp$loc)/3, 3)
temp$braingrid = array(temp$braingrid, c(3, temp$ndim))
temp$n.fiber = temp$n_fiber
temp$n.fiber2 = temp$n_fiber2

## run the tracking algorithm
nproj = 0  #skip zero voxels before termination (ROI I in HCP)
#nproj = 1  #skip  one voxel before termination (ROI II in HCP, 3D ROI simulation)
tobj <- v.track(v.obj=temp, xgrid.sp=temp$grid_sp[1], ygrid.sp=temp$grid_sp[2], zgrid.sp=temp$grid_sp[3], 
                braingrid=temp$braingrid, elim=T, nproj=nproj, vorient=c(1,1,1), elim.thres=5.5)
#tobj <- v.track(v.obj=temp, xgrid.sp=temp$grid_sp[1], ygrid.sp=temp$grid_sp[2], zgrid.sp=temp$grid_sp[3], 
#                braingrid=temp$braingrid, elim=T, nproj=nproj, vorient=c(1,1,1), elim.thres=6.5)

######################################
#### subregion fiber realizations ####
######################################
x_lim = c(0.5, temp$ndim[1]+0.5)
y_lim = c(0.5, temp$ndim[2]+0.5)
z_lim = c(0.5, temp$ndim[3]+0.5)

length(tobj$sorted.iinds[tobj$sorted.update.ind])
ndis <- length(tobj$sorted.iinds[tobj$sorted.update.ind])  # number of fibers
summary(tobj$lens[tobj$update.ind])  # summary of fiber lengths

connect.matrix <- function(tobj, num.subregion, braindim) {
  connect.mat <- matrix(0, num.subregion^3, num.subregion^3)
  for(i in which(tobj$update.ind)) {
    track.voxels <- c(tobj$tracks1[[i]]$voxels[length(tobj$tracks1[[i]]$voxels):2], tobj$tracks2[[i]]$voxels)
    for(j in 1:length(track.voxels)) {
      subvoxel <- (arrayInd(track.voxels[j], braindim)-1) %/% (braindim/num.subregion) + 1
      track.voxels[j] <- (subvoxel[3]-1)*num.subregion*num.subregion + (subvoxel[2]-1)*num.subregion + subvoxel[1]
    }
    track.voxels <- unique(track.voxels)
    if(length(track.voxels)>1) {
      for(j in 1:(length(track.voxels)-1)) {
        for(k in (j+1):length(track.voxels)) {
          connect.mat[track.voxels[j], track.voxels[k]] <- connect.mat[track.voxels[j], track.voxels[k]]+1
          connect.mat[track.voxels[k], track.voxels[j]] <- connect.mat[track.voxels[k], track.voxels[j]]+1
        }
      } 
    }
  }
  return(connect.mat)
}

num.subregion <- 3
braindim <- dim(temp$braingrid)[-1]
connect.mat.sm <- connect.matrix(tobj_sm, num.subregion, braindim)
connect.mat.sm.retest <- connect.matrix(tobj_sm_retest, num.subregion, braindim)
connect.mat.voxel <- connect.matrix(tobj_voxel, num.subregion, braindim)
connect.mat.voxel.retest <- connect.matrix(tobj_voxel_retest, num.subregion, braindim)

sum((connect.mat.sm-connect.mat.sm.retest)^2)
sum((connect.mat.voxel-connect.mat.voxel.retest)^2)

sqrt(sum((connect.mat.sm/sum(connect.mat.sm)-connect.mat.sm.retest/sum(connect.mat.sm.retest))^2))
sqrt(sum((connect.mat.voxel/sum(connect.mat.voxel)-connect.mat.voxel.retest/sum(connect.mat.voxel.retest))^2))

sum(abs(connect.mat.sm/sum(connect.mat.sm)-connect.mat.sm.retest/sum(connect.mat.sm.retest)))
sum(abs(connect.mat.voxel/sum(connect.mat.voxel)-connect.mat.voxel.retest/sum(connect.mat.voxel.retest)))

tmp = cbind(c(abs(connect.mat.sm/sum(connect.mat.sm)-connect.mat.sm.retest/sum(connect.mat.sm.retest))), c(abs(connect.mat.voxel/sum(connect.mat.voxel)-connect.mat.voxel.retest/sum(connect.mat.voxel.retest))))
boxplot(tmp)

tmp = cbind(c(abs(connect.mat.sm/sum(connect.mat.sm)-connect.mat.sm.retest/sum(connect.mat.sm.retest))/(connect.mat.sm/sum(connect.mat.sm)+connect.mat.sm.retest/sum(connect.mat.sm.retest))), c(abs(connect.mat.voxel/sum(connect.mat.voxel)-connect.mat.voxel.retest/sum(connect.mat.voxel.retest))/((connect.mat.voxel/sum(connect.mat.voxel)+connect.mat.voxel.retest/sum(connect.mat.voxel.retest)))))
boxplot(tmp)

tmp = cbind(c(abs(connect.mat.sm-connect.mat.sm.retest)/(connect.mat.sm+connect.mat.sm.retest)), c(abs(connect.mat.voxel-connect.mat.voxel.retest)/((connect.mat.voxel+connect.mat.voxel.retest))))
boxplot(tmp)

open3d()
for (iind in tobj$sorted.iinds[tobj$sorted.update.ind][1:1000]){
  cat(iind,"\n")
  # plot
  lineseg = t(t(tobj$tracks1[[iind]]$inloc)/temp$grid_sp+c(mean(x_lim),mean(y_lim),mean(z_lim)))
  plot.fib(lineseg, tobj$tracks1[[iind]]$dir,x_lim,y_lim,z_lim)
  lineseg = t(t(tobj$tracks2[[iind]]$inloc)/temp$grid_sp+c(mean(x_lim),mean(y_lim),mean(z_lim)))
  plot.fib(lineseg, tobj$tracks2[[iind]]$dir,x_lim,y_lim,z_lim)
}
decorate3d(xlim=range(x_lim), ylim=range(y_lim), zlim=range(z_lim), box = T, axis=T)
par3d(windowRect = c(0, 0, 700, 700))

for(view in 1:7) {
  load(paste0('view_hcp_',toString(view), '.Rdata'))
  rgl.viewpoint(scale=c(1,1,1),zoom=0.7,userMatrix = view_M)
  #rgl.viewpoint(scale=c(1,1,1),zoom=0.85,userMatrix = view_M)
  rgl.postscript(paste0('track_',name,'_view',toString(view),'.pdf'),fmt='pdf')
}

#rgl.close()

# create view matrix
# rotate figure in par3d and choose a suitable view angle
view <- 7
view_M <- par3d()$userMatrix
save(view_M,file=paste0('view_hcp_',toString(view),'.Rdata'))