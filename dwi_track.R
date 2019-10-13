# bi-directional fiber tracking for all fiber directions across all voxels
# vorient: if the voxel orientation does not match the physical orientation, make it -1
v.track <- function(v.obj, xgrid.sp, ygrid.sp, zgrid.sp, braingrid,
                    max.line=100, nproj=1, elim=T, elim.thres=1,
                    thres.ang=pi/6, vorient=rep(1,1,1))
{
  tracks1 <- list()
  tracks2 <- list()
  n.use.iind <- array(0, dim=length(v.obj$n.fiber2))  # number of tracts where each fiber direction appears
  n.iinds <- array(0,dim=length(v.obj$n.fiber2))  # number of voxels along tract starting from each fiber direction (bidirectional)
  lens <- array(0, dim=length(v.obj$n.fiber2))  # length of tract starting from each fiber direction (bidirectional)
  connect.mat <- array(0, dim=c(length(v.obj$n.fiber), length(v.obj$n.fiber)))
  
  for (iind in which(v.obj$n.fiber2>0)){
    cat(iind,"\n")
    tracks1[[iind]] <- fiber.track(iind=iind, eig=v.obj$vec, loc=v.obj$loc,
                                   map=v.obj$map, rmap=v.obj$rmap,
                                   n.fiber=v.obj$n.fiber, xgrid.sp=xgrid.sp,
                                   ygrid.sp=ygrid.sp, zgrid.sp=zgrid.sp, braingrid=braingrid,
                                   max.line=max.line, nproj=nproj, thres.ang=thres.ang,
                                   vorient=vorient)
    
    tracks2[[iind]] <- fiber.track(iind=iind, eig=-v.obj$vec, loc=v.obj$loc,
                                   map=v.obj$map, rmap=v.obj$rmap,
                                   n.fiber=v.obj$n.fiber, xgrid.sp=xgrid.sp,
                                   braingrid=braingrid,
                                   ygrid.sp=ygrid.sp, zgrid.sp=zgrid.sp,
                                   max.line=max.line, nproj=nproj, thres.ang=thres.ang,
                                   vorient=vorient)
    
    n.use.iind[tracks1[[iind]]$iinds] <- n.use.iind[tracks1[[iind]]$iinds] + 1
    n.use.iind[tracks2[[iind]]$iinds] <- n.use.iind[tracks2[[iind]]$iinds] + 1
    n.use.iind[iind] <- n.use.iind[iind] - 1
    n.iinds[iind] <- length(union(tracks1[[iind]]$iinds, tracks2[[iind]]$iinds))
    lens[iind] <- get.fdis(tracks1[[iind]]$inloc) + get.fdis(tracks2[[iind]]$inloc)
    for(i in 1:(length(tracks1[[iind]]$voxels)-1)) {
      connect.mat[tracks1[[iind]]$voxels[i], tracks1[[iind]]$voxels[i+1]] <- connect.mat[tracks1[[iind]]$voxels[i], tracks1[[iind]]$voxels[i+1]] + 1
      connect.mat[tracks1[[iind]]$voxels[i+1], tracks1[[iind]]$voxels[i]] <- connect.mat[tracks1[[iind]]$voxels[i+1], tracks1[[iind]]$voxels[i]] + 1
    }
    for(i in 1:(length(tracks2[[iind]]$voxels)-1)) {
      connect.mat[tracks2[[iind]]$voxels[i], tracks2[[iind]]$voxels[i+1]] <- connect.mat[tracks2[[iind]]$voxels[i], tracks2[[iind]]$voxels[i+1]] + 1
      connect.mat[tracks2[[iind]]$voxels[i+1], tracks2[[iind]]$voxels[i]] <- connect.mat[tracks2[[iind]]$voxels[i+1], tracks2[[iind]]$voxels[i]] + 1
    }
  }
  len.ord <- order(lens, decreasing=T)
  if (max(lens[n.iinds<=1])> elim.thres){
    cat("elim.thres is too small: it should be set at least", max(lens[n.iinds<=1]),"\n") 
  }
  
  # update.ind: set False if corresponding fiber direction is isotropic or has tract length smaller than elim.thres
  update.ind <- rep(T, length(v.obj$n.fiber2))
  update.ind[as.logical((v.obj$n.fiber2==0)+(lens<=elim.thres))] <- F
  
  sorted.iinds <- (1:length(v.obj$n.fiber2))[len.ord]  # indices of fiber direction sorted by tract length
  sorted.update.ind <- update.ind[len.ord]  # update.ind sorted by tract length
  
  return(list(tracks1=tracks1, tracks2=tracks2, n.iinds=n.iinds,
              n.use.iind=n.use.iind, update.ind=update.ind, sorted.iinds=sorted.iinds,
              sorted.update.ind=sorted.update.ind, lens=lens, connect.mat = connect.mat))
}


# plot fiber tracts
plot.fib <- function(loc, vec, x_lim, y_lim, z_lim) {
  
  loc1 <- matrix(rep(loc, each=2), nc=3)
  loc1 <- loc1[-c(1, nrow(loc1)), ]
  
  # get color
  vec <- abs(vec)
  if (length(vec) == 3) {
    vec <- matrix(vec, nc=3)
  }
  col1 <- rgb(red=vec[, 1], green=vec[, 2], blue=vec[, 3], alpha=1)
  col1 <- rep(col1, each=2)
  
  lines3d(loc1, lwd=1, col=col1)
}


# uni-directional fiber tracking for iind-th fiber direction
fiber.track <- function(iind, eig, loc, map, rmap, n.fiber, xgrid.sp, ygrid.sp,
                        zgrid.sp, braingrid, max.line=1000, nproj=1, thres.ang=pi/6,
                        vorient=c(1,1,1)) {
  
  braindim <- dim(braingrid)[-1]
  dimens <- c(xgrid.sp, ygrid.sp, zgrid.sp)

  path.voxel <- array(dim=max.line)  # indices of voxels along tract
  path.dir <- array(dim=c(max.line, 3))  # fiber directions along tract
  path.in <- array(dim=c(max.line, 3))  # fiber locations along tract
  path.change <- array(dim=max.line)  # indicators of whether fiber direction has been changed along tract
  path.iind <- array(dim=max.line)  # indices of fiber directions along tract

  # initialization of tract
  path.voxel[1] <- map[iind]  ## tract starting voxel
  path.dir[1,] <- eig[iind,]  ## tract starting direction
  path.in[1,] <- loc[iind,]   ## tract starting location
  path.change[1] <- T
  path.iind[1] <- iind  ## tract starting index (index of all fiber directions in the ROI)

  ii <- 1
  while (ii < max.line) {

    # determine outcoming location and outcoming voxel face in current voxel
    fio <- fiber.in.out(inc=path.in[ii,]-loc[path.iind[ii],], direct=path.dir[ii,], dimens=dimens)
    path.in[ii+1,] <- fio$outc + loc[path.iind[ii],]

    # determine the next voxel
    next.vox <- get.out.vox(fio$index, path.voxel[ii], braindim=braindim, vorient=vorient)

    if (is.na(next.vox)){
      break
    }

    # check whether next.vox has fiber directions within thres.ang towards fiber direction in current voxel; 
    # set change=True if yes; if change=False, check whether there exists any subsequent voxel along fiber direction 
    # in current voxel (limited number nproj) has fiber directions within thres.ang towards fiber direction in 
    # current voxel; set good=True if yes
    pro.res <- project.proceed(inc0=path.in[ii+1,], vox0=next.vox,
                               dir0=path.dir[ii,], loc, eig, rmap, n.fiber,
                               braindim, dimens, nproj=nproj,
                               thres.ang=thres.ang, vorient=vorient)
    change <- pro.res$first
    good <- pro.res$last

    if (!good){
      break
    }

    # update voxel and change
    path.voxel[ii+1] <- next.vox
    path.change[ii+1] <- change

    # update dir and iind
    # if change=True, record direction in next.vox (direction is changed)
    # if change=False, record direction in current voxel (direction remains the same)
    if (n.fiber[next.vox]<=1){
      path.iind[ii+1] <- rmap[next.vox]
      if (change){
        path.dir[ii+1,] <- eig[path.iind[ii+1],]
      } else {
        path.dir[ii+1,] <- path.dir[ii,]
      }
    } else {
      if (change){
        # choose fiber direction in next.vox which is the closest towards fiber direction in current voxel
        tiind <- rmap[next.vox]
        chosen <- which.max(abs(eig[tiind+(0:(n.fiber[next.vox]-1)),]%*%path.dir[ii,]))
        path.iind[ii+1] <- tiind+chosen-1
        path.dir[ii+1,] <- eig[path.iind[ii+1],]
      } else {
        path.iind[ii+1] <- rmap[next.vox]
        path.dir[ii+1,] <- path.dir[ii,]
      }
    }

    # align directions
    path.dir[ii+1,] <- sign(sum(path.dir[ii+1,]*path.dir[ii,]))*path.dir[ii+1,]

    ii <- ii+1
  }

  if (ii<max.line){
    path.in <- path.in[1:(ii+1),]
    path.iind <- path.iind[1:ii]
    path.dir <- path.dir[1:ii,]
    path.change <- path.change[1:ii]
    path.voxel <- path.voxel[1:ii]
  }
  return(list(inloc=path.in, dir=path.dir, iinds=path.iind, change=path.change, voxels=path.voxel))
}

# given fiber incoming location and fiber direction, determine outcoming location and ourcoming voxel face
fiber.in.out <- function(inc, direct, dimens){
  # assuming inc, outc are coordinates with the center of the voxel being (0,0,0)
  if (sum(dimens==0)){
    stop("directions has zero component, not yet supported! Please modify fiber.in.out\n")
  }
  # compute the distance of the current fiber directon to each face of the current voxel
  # create a box with x,y,z dimens and divide by travel speed(direct)
  tempdiff <- (round(cbind(dimens/2-inc,-inc-dimens/2),5)/direct)  # Hao: add round 5
  tempdiff[tempdiff==-Inf]=Inf  # Hao: added
  
  # get which axis is the current fiber direction hitting face of the current voxel first
  # 1:x  2:y  3:z
  index1 <- which.min(diag(tempdiff[,2-(direct>=0)]))  # Hao change direct>0 to direct>=0
  # which direction it is hitting 1:positive  2:negative
  index <- c(index1, (2-(direct>0))[index1])
  const <- tempdiff[index[1],index[2]]
  outc <- round(inc + const*direct,5)     ## Hao: add round5
  
  return(list(outc=outc, index=as.vector(index)))
}

get.out.vox <- function(index, cvox, braindim, vorient){
  # convert current voxel vector index to the 3d index in ROI
  cvoxindex <- as.vector(arrayInd(cvox, braindim))
  if (index[2]==1){
    # positive sides
    cvoxindex[index[1]] <- cvoxindex[index[1]] + vorient[index[1]]
  } else {
    # negative sides
    cvoxindex[index[1]] <- cvoxindex[index[1]] - vorient[index[1]]
  }
  if ((cvoxindex[index[1]]<1)||(cvoxindex[index[1]]>braindim[index[1]])){
    return(NA)
  } else {
    return((cvoxindex[3]-1)*braindim[1]*braindim[2] + (cvoxindex[2]-1)*braindim[1] + cvoxindex[1])
    #return(ArrayIndex(braindim, cvoxindex[1], cvoxindex[2], cvoxindex[3]))
  }
}

# check whether vox0 has fiber directions within thres.ang towards dir0
proceed <- function(vox0, dir0, eig, rmap, n.fiber, thres.ang=pi/6){
  good <- T
  if (n.fiber[vox0]==0){
    good <- F
  } else if (n.fiber[vox0]==1) {
    good <- acos(min(abs(eig[rmap[vox0],]%*%dir0),1))<thres.ang
  } else {
    good <- as.logical(sum(as.vector(acos(pmin(abs(eig[rmap[vox0]+(0:(n.fiber[vox0]-1)),]%*%dir0),1)))<thres.ang))
  }
  return(good)
}


project.proceed <- function(inc0, vox0, dir0, loc, eig, rmap, n.fiber,
                            braindim, dimens, nproj=2, thres.ang=pi/6,
                            vorient=c(1,1,1)){
  
  # check whether vox0 has fiber directions within thres.ang towards dir0; set first=True if yes
  first <- proceed(vox0, dir0, eig, rmap, n.fiber, thres.ang)

  # if first=False, check whether there exists any subsequent voxel along dir0 (limited number nproj) 
  # has fiber directions within thres.ang towards dir0; set last=True if yes
  num <- 1
  tinc <- inc0
  vox <- vox0
  last <- first
  iind <- rmap[vox]
  while ((!first) && (num <= nproj)){
    fio <- fiber.in.out(inc=tinc-loc[iind,], direct=dir0, dimens=dimens)
    tinc <- fio$outc + loc[iind,]

    # determine the next voxel
    vox <- get.out.vox(fio$index, vox, braindim=braindim, vorient=vorient)
    if (is.na(vox)){
      last <- F
      break
    }
    iind <- rmap[vox]
    # check whether vox has directions within thres.ang towards dir0
    last <- proceed(vox, dir0, eig, rmap, n.fiber, thres.ang)
    if (last){
      break
    }
    num <- num + 1
  }
  return(list(first=first, last=last))
}

#### get fiber distance ####
get.fdis <- function(inloc){
  if (nrow(inloc)==2){
    return(sum((apply(inloc,2,diff))^2))
  } else {
    return(sum(apply(apply(inloc,2,diff),1,function(x){sum(x^2)})))
  }
}